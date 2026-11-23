import argparse
import time
from pathlib import Path

import cv2


camera_index = 2
EXTERNAL_CAMERAS = {
    "top_5mp": ("2bdf", "028a"),      # 5MP USB Camera  P030D03  树顶正下俯视
    "side_1080p": ("2bdf", "0289"),   # 1080P USB Camera P030102 侧视四个挂点
}


def find_external_cameras() -> dict[str, str]:
    """Find the capture node (index 0) for each external USB camera."""
    cameras = {}
    for video in sorted(Path("/sys/class/video4linux").glob("video*")):
        if (video / "index").read_text().strip() != "0":
            continue
        resolved = video.resolve()
        for parent in (resolved, *resolved.parents):
            vendor = parent / "idVendor"
            product = parent / "idProduct"
            if vendor.exists() and product.exists():
                identity = (vendor.read_text().strip(), product.read_text().strip())
                for name, expected in EXTERNAL_CAMERAS.items():
                    if identity == expected:
                        if name in cameras:
                            raise RuntimeError(f"[Camera] 检测到多台 {name}，无法确定使用哪一台")
                        cameras[name] = f"/dev/{video.name}"
                break

    missing = EXTERNAL_CAMERAS.keys() - cameras.keys()
    if missing:
        raise RuntimeError(f"[Camera] 未找到外接相机: {', '.join(sorted(missing))}")
    return cameras


def init_camera(index: int | str | None = None, width: int = 1920, height: int = 1080,
                fps: int = 30) -> cv2.VideoCapture:
    if index is None:
        index = camera_index
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"[Camera] 无法打开相机 {index}")

    # 按顺序设置 OpenCV 属性
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    device = f"/dev/video{index}" if isinstance(index, int) else index
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    
    # 其他图像参数
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, 32)
    cap.set(cv2.CAP_PROP_SATURATION, 64)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.set(cv2.CAP_PROP_SHARPNESS, 10)
    cap.set(cv2.CAP_PROP_GAMMA, 100)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    for _ in range(10):
        cap.grab()

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Camera] {device} 已初始化: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(actual_fps)}fps")
          
    return cap


def init_external_cameras(width: int = 1920, height: int = 1080,
                          fps: int = 30) -> dict[str, cv2.VideoCapture]:
    """Open both external cameras; release the first if the second fails.

    相机的分辨率/帧率/图像参数只在这里决定，别处不要再调 cap.set。
    """
    devices = find_external_cameras()
    cameras = {}
    try:
        for name, device in devices.items():
            cameras[name] = init_camera(device, width, height, fps)
    except Exception:
        for cap in cameras.values():
            cap.release()
        raise
    return cameras


def draw_overlay(frame, name: str, fps: float, scale: float):
    """缩放到 scale，并画上名称/分辨率/帧率、中心十字与中央 1/4 参考框。"""
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        frame = frame.copy()
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 24, cy), (cx + 24, cy), (0, 255, 255), 1)
    cv2.line(frame, (cx, cy - 24), (cx, cy + 24), (0, 255, 255), 1)
    # 叠加层颜色必须落在真实物体的色域之外：奶黄果是 h13-17/s28-72，而黄 (0,200,200) 的
    # HSV 是 h30/s255/v200，正好落在奶黄色域里 —— 截图拿去标定时会整框被当成一颗巨果。
    cv2.rectangle(frame, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 0, 255), 1)
    cv2.putText(frame, f"{name}  {int(w / scale)}x{int(h / scale)}  {fps:5.1f} fps", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "q/ESC quit   s snapshot   +/- scale", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def preview_external_cameras(scale: float = 0.5, snapshot_dir: str = "debug/cams") -> None:
    """两台相机实时同时预览：q/ESC 退出，s 存原始帧，+/- 缩放。"""
    cameras = init_external_cameras()
    snapshot = Path(snapshot_dir)
    snapshot.mkdir(parents=True, exist_ok=True)
    prev = {name: 0.0 for name in cameras}
    fps = {name: 0.0 for name in cameras}
    raw: dict[str, object] = {}
    try:
        for i, name in enumerate(cameras):
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(name, 40 + i * 680, 40)
        while True:
            for name, cap in cameras.items():
                if not cap.grab():
                    raise RuntimeError(f"[Camera] {name} 取帧失败")
            now = time.perf_counter()
            for name, cap in cameras.items():
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError(f"[Camera] {name} 解码失败")
                if prev[name]:
                    inst = 1.0 / max(now - prev[name], 1e-6)
                    fps[name] = inst if not fps[name] else 0.9 * fps[name] + 0.1 * inst
                prev[name] = now
                raw[name] = frame
                cv2.imshow(name, draw_overlay(frame, name, fps[name], scale))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("="), ord("+")):
                scale = min(round(scale + 0.25, 2), 1.0)
            elif key == ord("-"):
                scale = max(round(scale - 0.25, 2), 0.25)
            elif key == ord("s"):
                stamp = time.strftime("%m%d_%H%M%S")
                for name, frame in raw.items():
                    path = snapshot / f"{name}_{stamp}.png"
                    cv2.imwrite(str(path), frame)
                    print(f"[Camera] {name} 已存 {path}")
    finally:
        for cap in cameras.values():
            cap.release()
        cv2.destroyAllWindows()


def save_frames(directory: str, count: int = 3) -> None:
    """无窗口抓帧落盘：无显示器的环境（SSH / 远程）用它验证两台相机能否同时工作。"""
    cameras = init_external_cameras()
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    try:
        for name, cap in cameras.items():
            for i in range(count):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"[Camera] {name} 取帧失败")
                path = out / f"{name}_{i}.png"
                cv2.imwrite(str(path), frame)
                print(f"[Camera] {name} -> {path}  {frame.shape[1]}×{frame.shape[0]}")
    finally:
        for cap in cameras.values():
            cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="同时预览两台外接 USB 摄像头")
    parser.add_argument("--list", action="store_true", help="只显示识别到的设备节点")
    parser.add_argument("--save", metavar="DIR", help="抓几帧存到 DIR，不弹窗口")
    parser.add_argument("--count", type=int, default=3, help="--save 时每台相机的帧数")
    parser.add_argument("--scale", type=float, default=0.5, help="预览窗口缩放（默认 0.5）")
    parser.add_argument("--snapshot-dir", default="debug/cams", help="预览时按 s 存图的目录")
    args = parser.parse_args()
    if args.list:
        for name, device in find_external_cameras().items():
            print(f"{name}: {device}")
    elif args.save:
        save_frames(args.save, args.count)
    else:
        preview_external_cameras(args.scale, args.snapshot_dir)
