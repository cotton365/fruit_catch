from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TopDetection:
    found: bool
    target_center: tuple[float, float] | None
    reference_center: tuple[float, float]
    dx_px: float | None
    dy_px: float | None
    component_area_px: int
    component_box: tuple[int, int, int, int] | None
    orientation_deg: float | None


@dataclass(frozen=True)
class BlobInfo:
    area_px: float
    fill_ratio: float
    elongation: float
    circularity: float
    center: tuple[float, float]


@dataclass(frozen=True)
class SideDetection:
    green_positions: tuple[str, ...]
    blobs: dict[str, tuple[BlobInfo, ...]]


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _green_mask(frame: np.ndarray, lower: list[int], upper: list[int]) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def _occupancy(
    mask: np.ndarray,
    center: tuple[float, float],
    radius: float,
    annulus_scale: list[float] | tuple[float, float],
) -> tuple[float, float]:
    """返回 (果盘内掩码占比, 果盘外环带掩码占比)。

    环带内径/外径都是 radius 的倍数。果盘内的占比回答"这里有没有一块果形的东西"，
    果盘与环带之差回答"这块东西是不是独立凸出来的"——树干上下都是它自己，差值会被拉低。
    """
    height, width = mask.shape[:2]
    cx, cy = center
    inner = radius * float(annulus_scale[0])
    outer = radius * float(annulus_scale[1])
    x1, x2 = max(0, int(cx - outer)), min(width, int(cx + outer) + 1)
    y1, y2 = max(0, int(cy - outer)), min(height, int(cy + outer) + 1)
    if x1 >= x2 or y1 >= y2:
        return 0.0, 0.0
    yy, xx = np.mgrid[y1:y2, x1:x2]
    distance2 = (xx - cx) ** 2 + (yy - cy) ** 2
    patch = mask[y1:y2, x1:x2].astype(bool)
    disk = distance2 <= radius * radius
    ring = (distance2 > inner * inner) & (distance2 <= outer * outer)
    disk_fill = float(patch[disk].mean()) if disk.any() else 0.0
    ring_fill = float(patch[ring].mean()) if ring.any() else 0.0
    return disk_fill, ring_fill


def _pixel_roi(shape: tuple[int, ...], roi: list[float]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1 = int(np.clip(roi[0], 0.0, 1.0) * width)
    y1 = int(np.clip(roi[1], 0.0, 1.0) * height)
    x2 = int(np.clip(roi[2], 0.0, 1.0) * width)
    y2 = int(np.clip(roi[3], 0.0, 1.0) * height)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


class TopCenterDetector:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def detect(self, frame: np.ndarray) -> tuple[TopDetection, np.ndarray]:
        height, width = frame.shape[:2]
        mask = _green_mask(
            frame,
            self.config["green_hsv_lower"],
            self.config["green_hsv_upper"],
        )
        x1, y1, x2, y2 = _pixel_roi(frame.shape, self.config["roi"])
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        open_size = _odd_kernel(self.config.get("open_kernel_px", 5))
        close_size = _odd_kernel(self.config.get("close_kernel_px", 13))
        roi_mask = cv2.morphologyEx(
            roi_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
        )
        roi_mask = cv2.morphologyEx(
            roi_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask, 8)
        min_area = int(self.config.get("min_component_area_px", 3000))
        min_width = int(width * float(self.config.get("min_component_width_ratio", 0.0)))
        min_height = int(height * float(self.config.get("min_component_height_ratio", 0.0)))
        candidates = [
            index
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= min_area
            and stats[index, cv2.CC_STAT_WIDTH] >= min_width
            and stats[index, cv2.CC_STAT_HEIGHT] >= min_height
        ]

        offset_x, offset_y = self.config.get("reference_offset_px", [0, 0])
        reference = (width / 2.0 + float(offset_x), height / 2.0 + float(offset_y))
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.drawMarker(
            annotated,
            (int(round(reference[0])), int(round(reference[1]))),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            28,
            2,
        )

        if not candidates:
            cv2.putText(annotated, "TREE NOT FOUND", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return TopDetection(False, None, reference, None, None, 0, None, None), annotated

        component = max(candidates, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        box_width = int(stats[component, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        area = int(stats[component, cv2.CC_STAT_AREA])

        # The bounding-box center is less sensitive than a raw centroid to a single fruit.
        target = (x + box_width / 2.0, y + box_height / 2.0)
        dx = target[0] - reference[0]
        dy = target[1] - reference[1]

        points_yx = np.column_stack(np.where(labels == component)).astype(np.float32)
        orientation_deg: float | None = None
        if len(points_yx) >= 20:
            points_xy = points_yx[:, ::-1]
            centered = points_xy - points_xy.mean(axis=0, keepdims=True)
            covariance = centered.T @ centered / float(len(points_xy))
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            direction = eigenvectors[:, int(np.argmax(eigenvalues))]
            angle = float(np.degrees(np.arctan2(-direction[1], direction[0])))
            while angle >= 90.0:
                angle -= 180.0
            while angle < -90.0:
                angle += 180.0
            orientation_deg = angle

        component_mask = np.where(labels == component, 255, 0).astype(np.uint8)
        overlay = np.zeros_like(annotated)
        overlay[:, :, 1] = component_mask
        annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.25, 0.0)
        cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), (0, 255, 0), 2)
        cv2.drawMarker(
            annotated,
            (int(round(target[0])), int(round(target[1]))),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            28,
            2,
        )
        cv2.line(
            annotated,
            (int(round(reference[0])), int(round(reference[1]))),
            (int(round(target[0])), int(round(target[1]))),
            (255, 255, 0),
            2,
        )
        if orientation_deg is not None:
            direction_angle = np.radians(-orientation_deg)
            line_length = max(40.0, 0.35 * min(box_width, box_height))
            line_dx = float(np.cos(direction_angle) * line_length)
            line_dy = float(-np.sin(direction_angle) * line_length)
            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
            cv2.line(
                annotated,
                (int(round(center_x - line_dx)), int(round(center_y - line_dy))),
                (int(round(center_x + line_dx)), int(round(center_y + line_dy))),
                (255, 0, 255),
                3,
            )
        cv2.putText(
            annotated,
            f"dx={dx:+.1f}px dy={dy:+.1f}px",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        if orientation_deg is not None:
            cv2.putText(
                annotated,
                f"angle={orientation_deg:+.1f} deg",
                (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 255),
                2,
            )
        return TopDetection(
            True,
            target,
            reference,
            dx,
            dy,
            area,
            (x, y, box_width, box_height),
            orientation_deg,
        ), annotated


class SideColorDetector:
    POSITION_ORDER = ("L1", "L2", "R1", "R2")

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def detect(self, frame: np.ndarray) -> tuple[SideDetection, np.ndarray]:
        mask = _green_mask(
            frame,
            self.config["green_hsv_lower"],
            self.config["green_hsv_upper"],
        )
        # 开运算的圆盘必须比挂销粗、比果细，否则果会和枝干连成一个连通域被整块丢掉。
        fruit_radius = float(self.config.get("fruit_radius_px", 55))
        open_size = _odd_kernel(
            int(round(2.0 * fruit_radius * float(self.config.get("open_radius_ratio", 0.38)) + 1))
        )
        close_size = _odd_kernel(self.config.get("close_kernel_px", 5))
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
        )

        min_area = float(self.config.get("min_blob_area_px", 350))
        max_area = float(self.config.get("max_blob_area_px", 30000))
        min_fill = float(self.config.get("min_fill_ratio", 0.35))
        max_elongation = float(self.config.get("max_elongation", 2.2))
        min_circularity = float(self.config.get("min_circularity", 0.18))
        min_disk_fill = float(self.config.get("min_disk_fill", 0.70))
        min_protrusion = float(self.config.get("min_protrusion", 0.65))
        annulus_scale = self.config.get("annulus_scale", [1.2, 1.6])

        annotated = frame.copy()
        position_blobs: dict[str, list[BlobInfo]] = {position: [] for position in self.POSITION_ORDER}

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        unowned: list[tuple[float, float]] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            rectangle = cv2.minAreaRect(contour)
            rect_width, rect_height = rectangle[1]
            short_side = max(1.0, min(rect_width, rect_height))
            long_side = max(rect_width, rect_height)
            elongation = long_side / short_side
            _, _, bounding_width, bounding_height = cv2.boundingRect(contour)
            fill_ratio = area / max(1.0, float(bounding_width * bounding_height))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4.0 * np.pi * area / max(1.0, perimeter * perimeter)
            if elongation > max_elongation or fill_ratio < min_fill or circularity < min_circularity:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            center = (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
            disk_fill, ring_fill = _occupancy(mask, center, fruit_radius, annulus_scale)
            if disk_fill < min_disk_fill or disk_fill - ring_fill < min_protrusion:
                continue
            owner = self._owner(frame.shape, center)
            if owner is None:
                unowned.append(center)
                continue
            position_blobs[owner].append(BlobInfo(area, fill_ratio, elongation, circularity, center))

        detected_positions = [position for position in self.POSITION_ORDER if position_blobs[position]]
        for position in self.POSITION_ORDER:
            roi = self.config["positions"][position]
            x1, y1, x2, y2 = _pixel_roi(frame.shape, roi)
            is_green = bool(position_blobs[position])
            color = (0, 255, 0) if is_green else (90, 90, 90)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                position,
                (x1 + 6, max(24, y1 + 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            for blob in position_blobs[position]:
                cv2.circle(annotated, (int(blob.center[0]), int(blob.center[1])), 10, (0, 255, 255), 2)
        for center in unowned:
            cv2.circle(annotated, (int(center[0]), int(center[1])), 10, (0, 128, 255), 2)

        frozen: dict[str, tuple[BlobInfo, ...]] = {
            position: tuple(blobs) for position, blobs in position_blobs.items()
        }
        return SideDetection(tuple(detected_positions), frozen), annotated

    def _owner(self, shape: tuple[int, ...], center: tuple[float, float]) -> str | None:
        cx, cy = center
        boxes = {
            position: _pixel_roi(shape, self.config["positions"][position])
            for position in self.POSITION_ORDER
        }
        containing = [
            position
            for position, (x1, y1, x2, y2) in boxes.items()
            if x1 <= cx <= x2 and y1 <= cy <= y2
        ]
        if not containing:
            return None
        if len(containing) == 1:
            return containing[0]

        def center_distance(position: str) -> float:
            x1, y1, x2, y2 = boxes[position]
            return (cx - (x1 + x2) / 2.0) ** 2 + (cy - (y1 + y2) / 2.0) ** 2

        return min(containing, key=center_distance)
