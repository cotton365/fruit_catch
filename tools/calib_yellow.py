#!/usr/bin/env python3
"""黄果标定工具：实时看 + 把"奶黄果"的帧变成"位置 → 外接四边形 / 中心 / 果径"的表。

黄果与树不同色，所以它能干净地被分割出来；而"某一档 yaw、某个挂点，果在画面里的
坐标和大小"是纯几何量，与果子颜色无关。因此这表的用途是把绿果的检测窗口和尺度定下来。

用法：
    python3 tools/calib_yellow.py --live                      # 常开窗口，实时看两台相机
    python3 tools/calib_yellow.py                            # 处理 debug/cams/taken 里的帧
    python3 tools/calib_yellow.py debug/cams/_side_cam.png    # 单张
    python3 tools/calib_yellow.py debug/cams/taken/           # 整目录（会额外给包络表）

实时模式（--live）：
    相机初始化、分辨率/帧率/曝光全部由 camera_init2.init_external_cameras() 决定，本文件
    不碰任何 cap.set。原始帧是相机给出的尺寸（现在 init 里写的是 1920x1080），窗口里显示
    的那份按 --window-width 缩小，按 `s` 存的是**原始尺寸、无叠加层**的 PNG。
    按键：q/ESC 退出，s 存图 + 打印精确表格（含 R_core）。

输入请用预览里按 `s` 存的**原始帧**（不含叠加层）。截图也能跑——工具会把叠加层颜色
剥掉——但那只是兜底；二值化不应该依赖"框画成什么颜色"，靠的是原始帧本来就没有框。

输出（都写到 --out-dir）：
    <名字>_mask.png   奶黄二值图，与表格完全同源（剥叠加层 → 开运算剪细线 → 闭运算）
    <名字>_calib.png  标注图：四窗口 + 果的外接四边形 + 内接圆
    表格              每个候选黄果一行；整目录时额外给每个位置的 min/max 包络
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TOOLS_DIR))
from vision.detectors import _green_mask, _pixel_roi  # noqa: E402

# 实测（第 1 组真图）：奶黄果核心 h13-17 s28-72 v119-167；树干 h71；地面 h19 s23 v87。
# s 上限 120 不是随手写的：奶黄漆本来就淡，而叠加层参考框的 s 高达 255，这一条把两者彻底分开。
CREAM_HSV_LOWER = [5, 45, 110]
CREAM_HSV_UPPER = [32, 120, 255]

# camera_init2.py 叠加层的合成色（真帧没有这些颜色）。参考框原先用 (0,200,200)，其 HSV 是
# h30 s255 v200 —— 正好落在 CREAM 区间内，会把整棵树框成"一个巨大的果子"，故已改为红色
# (0,0,255)。红色本身不落进 CREAM，但截图缩放/压缩产生的 20% 混合像素仍可能擦边（实测
# (96,112,179) = h6 s118 v179），所以这里把新红版也一并剥掉，作为第二道保险。
OVERLAY_COLORS = [(0, 255, 0), (0, 255, 255), (0, 200, 200), (0, 0, 255)]
OVERLAY_TOLERANCE = 45

REPORT_HEADER = (
    f"{'文件':22s}{'窗口':>7s}{'bbox x,y,w,h':>22s}{'中心':>16s}{'面积':>8s}"
    f"{'fill':>6s}{'elong':>6s}{'直径':>6s}{'R':>5s}{'R_core':>7s}"
)


def strip_overlay(frame: np.ndarray, tolerance: int = OVERLAY_TOLERANCE) -> np.ndarray:
    """把叠加层颜色涂黑（黑不在 CREAM 区间内，v=0 < 110），截图与真帧都能用。"""
    keep = np.ones(frame.shape[:2], dtype=bool)
    for b, g, r in OVERLAY_COLORS:
        keep &= ~(
            (np.abs(frame[:, :, 0].astype(np.int16) - b) <= tolerance)
            & (np.abs(frame[:, :, 1].astype(np.int16) - g) <= tolerance)
            & (np.abs(frame[:, :, 2].astype(np.int16) - r) <= tolerance)
        )
    stripped = frame.copy()
    stripped[~keep] = 0
    return stripped


def load_frame(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)  # 兼容中文文件名
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"读不出图像：{path}")
    return frame


def find_cream_blobs(
    frame: np.ndarray, min_area: float, strip: bool = True, need_core: bool = True
) -> tuple[np.ndarray, list[dict]]:
    if strip:
        frame = strip_overlay(frame)
    mask = _green_mask(frame, CREAM_HSV_LOWER, CREAM_HSV_UPPER)
    # 先开掉细线（叠加层参考框只有 1-2px 粗，会像尾巴一样挂在果上把 bbox 撑大），再闭合成块。
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # 实时窗口每帧都算内接半径太贵（1080p 的 distanceTransform），只在出表/存图时算。
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5) if need_core else None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[dict] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w * h <= 0:
            continue
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)
        short_side = max(1.0, min(side_a, side_b))
        long_side = max(side_a, side_b)
        perimeter = float(cv2.arcLength(contour, True))
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        blobs.append(
            {
                "box": (x, y, w, h),
                "center": (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
                "area": area,
                "fill": area / float(w * h),
                "elongation": long_side / short_side,
                "circularity": 4.0 * math.pi * area / max(1.0, perimeter * perimeter),
                "diameter": math.sqrt(4.0 * area / math.pi),
                "radius": math.sqrt(area / math.pi),
                # 内切半径：果子上最大内接圆的半径。背景/边框粘连只会让 bbox 和面积变大，
                # 不会让这个数变大，所以 fruit_radius_px 应该用它而不是 sqrt(A/π)。
                "core_radius": float(distance[y : y + h, x : x + w].max()) if distance is not None else float("nan"),
            }
        )
    blobs.sort(key=lambda b: b["area"], reverse=True)
    return mask, blobs


def containing_windows(shape: tuple[int, ...], positions: dict[str, list[float]], center) -> list[str]:
    hits = []
    for position, roi in positions.items():
        x1, y1, x2, y2 = _pixel_roi(shape, roi)
        if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
            hits.append(position)
    return sorted(hits, key=lambda p: list(positions).index(p))


def format_row(label: str, blob: dict, windows: list[str]) -> str:
    cx, cy = blob["center"]
    x, y, w, h = blob["box"]
    core = blob["core_radius"]
    core_text = f"{core:.1f}" if math.isfinite(core) else "—"
    return (
        f"{label:22s}{'/'.join(windows) or '框外':>7s}"
        f"{f'{x},{y},{w},{h}':>22s}{f'{cx:.0f},{cy:.0f}':>16s}{blob['area']:8.0f}"
        f"{blob['fill']:6.2f}{blob['elongation']:6.2f}{blob['diameter']:6.0f}"
        f"{blob['radius']:5.1f}{core_text:>7s}"
    )


def annotate(frame: np.ndarray, rows: list[dict], positions: dict[str, list[float]]) -> np.ndarray:
    canvas = frame.copy()
    for position, roi in positions.items():
        x1, y1, x2, y2 = _pixel_roi(frame.shape, roi)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (140, 140, 140), 1)
        cv2.putText(canvas, position, (x1 + 6, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
    for row in rows:
        x, y, w, h = row["box"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cx, cy = row["center"]
        cv2.circle(canvas, (int(cx), int(cy)), int(round(row["radius"])), (0, 255, 255), 2)
        cv2.putText(
            canvas,
            f"{'/'.join(row['windows']) or '框外'} d={row['diameter']:.0f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )
    return canvas


def report(
    frames: list[Path], positions: dict[str, list[float]], min_area: float, out_dir: Path | None, strip: bool
) -> None:
    per_position: dict[str, list[dict]] = {position: [] for position in positions}
    print(REPORT_HEADER)
    for path in frames:
        frame = load_frame(path)
        mask, blobs = find_cream_blobs(frame, min_area, strip)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            # 空位帧也要存二值图：那是"为什么什么都没检出"的唯一证据。
            cv2.imwrite(str(out_dir / f"{path.stem}_mask.png"), mask)
        if not blobs:
            print(f"{path.name:22s}{'—':>7s}  无奶黄果（空位基准 / 或这帧没挂黄果）")
            continue
        rows = []
        for blob in blobs:
            windows = containing_windows(frame.shape, positions, blob["center"])
            rows.append({**blob, "windows": windows})
            if len(windows) == 1:
                per_position[windows[0]].append(blob)
            print(format_row(path.name, blob, windows))
        if out_dir is not None:
            cv2.imwrite(str(out_dir / f"{path.stem}_calib.png"), annotate(frame, rows, positions))

    if len(frames) > 1:
        print("\n每个位置的包络（同一位置多帧的 min/max，含果子自转造成的表观变化）：")
        print(
            f"{'位置':>6s}{'帧数':>6s}{'宽 min..max':>14s}{'高 min..max':>14s}"
            f"{'面积 min..max':>18s}{'R_core min..max':>17s}"
        )
        for position, blobs in per_position.items():
            if not blobs:
                print(f"{position:>6s}{0:6d}   —（这组帧里这个位置没有可用的单窗黄果）")
                continue
            widths = [b["box"][2] for b in blobs]
            heights = [b["box"][3] for b in blobs]
            areas = [b["area"] for b in blobs]
            core = [b["core_radius"] for b in blobs if math.isfinite(b["core_radius"])]
            core_text = f"{min(core):.1f}..{max(core):.1f}" if core else "—"
            print(
                f"{position:>6s}{len(blobs):6d}{f'{min(widths)}..{max(widths)}':>14s}"
                f"{f'{min(heights)}..{max(heights)}':>14s}{f'{min(areas):.0f}..{max(areas):.0f}':>18s}"
                f"{core_text:>17s}"
            )
        print("\n注意：多窗命中的果没有计入包络（它落在框重叠区，本表的裁决还不唯一）。")


def draw_live(
    canvas: np.ndarray, shape: tuple[int, ...], blobs: list[dict], positions: dict[str, list[float]], scale: float
) -> np.ndarray:
    """在缩小后的画布上叠四窗口与奶黄果的外接四边形（坐标按 scale 换算，不缩放原帧）。"""
    for position, roi in positions.items():
        x1, y1, x2, y2 = _pixel_roi(shape, roi)
        a = (int(round(x1 * scale)), int(round(y1 * scale)))
        b = (int(round(x2 * scale)), int(round(y2 * scale)))
        cv2.rectangle(canvas, a, b, (140, 140, 140), 1)
        cv2.putText(canvas, position, (a[0] + 5, a[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
    for blob in blobs:
        x, y, w, h = blob["box"]
        a = (int(round(x * scale)), int(round(y * scale)))
        b = (int(round((x + w) * scale)), int(round((y + h) * scale)))
        cv2.rectangle(canvas, a, b, (0, 200, 255), 2)
        windows = containing_windows(shape, positions, blob["center"])
        label = f"{'/'.join(windows) or '框外'} A={blob['area']:.0f} d={blob['diameter']:.0f}"
        cv2.putText(
            canvas, label, (a[0], max(16, a[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2
        )
    return canvas


def _snapshot(
    frames: dict[str, np.ndarray],
    positions: dict[str, list[float]],
    min_area: float,
    strip: bool,
    snapshot_dir: Path,
) -> None:
    stamp = time.strftime("%m%d_%H%M%S")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{stamp}] 存原始帧 + 精确表格")
    print(REPORT_HEADER)
    for name, frame in frames.items():
        path = snapshot_dir / f"{name}_{stamp}.png"
        cv2.imwrite(str(path), frame)  # 原始尺寸、无叠加层
        mask, blobs = find_cream_blobs(frame, min_area, strip, need_core=True)
        cv2.imwrite(str(snapshot_dir / f"{name}_{stamp}_mask.png"), mask)
        print(f"{path.name:22s}  {frame.shape[1]}x{frame.shape[0]}")
        if not blobs:
            print(f"{'':22s}{'—':>7s}  无奶黄果（空位基准 / 或这帧没挂黄果）")
            continue
        for blob in blobs:
            print(format_row(path.name, blob, containing_windows(frame.shape, positions, blob["center"])))
    print(f"帧与二值图已写到 {snapshot_dir}/")


def live_view(
    positions: dict[str, list[float]], min_area: float, window_width: int, snapshot_dir: Path, strip: bool
) -> None:
    """两台相机常开实时窗口。相机参数全部由 camera_init2 决定，本函数不碰 cap.set。"""
    try:
        import camera_init2 as preview  # 与本文件同目录：相机初始化 + HUD 都复用它
    except ImportError as exc:  # pragma: no cover - 缺文件时才走
        raise SystemExit(f"实时模式需要 tools/camera_init2.py：{exc}") from exc

    cameras = preview.init_external_cameras()
    windows = {}
    fps_ema: dict[str, float] = {}
    last: dict[str, float] = {}
    sized: set[str] = set()
    try:
        for index, name in enumerate(cameras):
            window = f"calib {name}"
            windows[name] = window
            fps_ema[name] = 0.0
            last[name] = time.perf_counter()
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.moveWindow(window, 40 + index * 60, 40 + index * 60)
        print("实时窗口已启动：q/ESC 退出，s 存原始帧 + 打印表格。")
        print(f"显示宽度 {window_width}px；存图目录 {snapshot_dir}/")
        while True:
            frames: dict[str, np.ndarray] = {}
            for cap in cameras.values():
                cap.grab()
            for name, cap in cameras.items():
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                frames[name] = frame
                now = time.perf_counter()
                delta = now - last[name]
                last[name] = now
                if delta > 0:
                    fps_ema[name] = (1.0 / delta) if fps_ema[name] == 0.0 else 0.9 * fps_ema[name] + 0.1 * (1.0 / delta)
                scale = min(1.0, window_width / float(frame.shape[1]))
                canvas = preview.draw_overlay(frame, name, fps_ema[name], scale)
                _, blobs = find_cream_blobs(frame, min_area, strip, need_core=False)
                draw_live(canvas, frame.shape, blobs, positions, scale)
                cv2.imshow(windows[name], canvas)
                if name not in sized:
                    cv2.resizeWindow(windows[name], canvas.shape[1], canvas.shape[0])
                    sized.add(name)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and frames:
                _snapshot(frames, positions, min_area, strip, snapshot_dir)
    finally:
        for cap in cameras.values():
            cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="黄果标定：实时看 + 帧 → 位置/外接四边形/果径 表")
    parser.add_argument("paths", nargs="*", help="图像文件或目录；缺省处理 debug/cams/taken")
    parser.add_argument("--live", action="store_true", help="常开实时窗口（q/ESC 退出，s 存图）")
    parser.add_argument("--window-width", type=int, default=960, help="实时窗口显示宽度（默认 960，原帧不缩放）")
    parser.add_argument("--snapshot-dir", type=Path, default=REPO_ROOT / "debug/cams/taken", help="实时模式存图目录")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.json", help="取 positions 四框")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "debug/cams/calib", help="标注图输出目录；传 '' 关闭"
    )
    parser.add_argument("--min-area", type=float, default=1500.0, help="候选最小面积（px²）")
    parser.add_argument(
        "--no-strip-overlay", action="store_true", help="不剥掉 camera_init2 叠加层颜色（真帧无需剥）"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    positions = config["side_detector"]["positions"]
    strip = not args.no_strip_overlay

    if args.live:
        live_view(positions, args.min_area, args.window_width, Path(args.snapshot_dir), strip)
        return

    frames: list[Path] = []
    missing: list[str] = []
    for raw in args.paths or [str(REPO_ROOT / "debug/cams/taken")]:
        path = Path(raw)
        if path.is_dir():
            for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                frames.extend(sorted(path.glob(suffix)))
        elif path.exists():
            frames.append(path)
        else:
            missing.append(str(path))
    # 跳过本工具自己的产物，避免把 _mask/_calib 再当输入吃一遍。
    frames = [f for f in frames if not f.stem.endswith(("_mask", "_calib"))]
    if not frames:
        raise SystemExit(
            f"没有可读的图像（{', '.join(missing) if missing else '目录里没有图像'}）：先跑 --live 按 s 存帧，"
            "或把图像路径作为参数传进来"
        )

    out_dir = Path(args.out_dir) if str(args.out_dir) else None
    report(frames, positions, args.min_area, out_dir, strip)
    if out_dir is not None:
        print(f"\n二值图 <名字>_mask.png 与标注图 <名字>_calib.png 已写到 {out_dir}/")


if __name__ == "__main__":
    main()
