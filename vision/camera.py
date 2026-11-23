from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any

import cv2


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraSpec:
    name: str
    device: str
    width: int
    height: int
    fps: int
    rotate: int
    flip_horizontal: bool
    flip_vertical: bool
    warmup_frames: int
    controls: dict[str, int]
    lock_controls: dict[str, int]

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "CameraSpec":
        return cls(
            name=name,
            device=str(config["device"]),
            width=int(config.get("width", 1280)),
            height=int(config.get("height", 720)),
            fps=int(config.get("fps", 30)),
            rotate=int(config.get("rotate", 0)) % 360,
            flip_horizontal=bool(config.get("flip_horizontal", False)),
            flip_vertical=bool(config.get("flip_vertical", False)),
            warmup_frames=int(config.get("warmup_frames", 8)),
            controls={str(k): int(v) for k, v in config.get("controls", {}).items()},
            lock_controls={str(k): int(v) for k, v in config.get("lock_controls", {}).items()},
        )


def apply_v4l2_controls(spec: CameraSpec, controls: dict[str, int]) -> None:
    if not controls:
        return
    assignments = ",".join(f"{key}={value}" for key, value in controls.items())
    result = subprocess.run(
        ["v4l2-ctl", "-d", spec.device, "--set-ctrl", assignments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise CameraError(f"Failed to configure {spec.name} camera: {message}")


class UsbCamera:
    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.close()
        if not os.path.exists(self.spec.device):
            raise CameraError(f"Camera path does not exist: {self.spec.device}")

        apply_v4l2_controls(self.spec, self.spec.controls)
        capture = cv2.VideoCapture(self.spec.device, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        capture.set(cv2.CAP_PROP_FPS, self.spec.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not capture.isOpened():
            capture.release()
            raise CameraError(f"Cannot open {self.spec.name} camera: {self.spec.device}")

        frame = None
        for _ in range(max(1, self.spec.warmup_frames + 1)):
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                raise CameraError(f"Cannot read a warmup frame from {self.spec.name} camera")

        # Let automatic white balance settle, then freeze it for repeatable HSV colors.
        apply_v4l2_controls(self.spec, self.spec.lock_controls)
        for _ in range(2):
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                raise CameraError(f"Cannot read a locked frame from {self.spec.name} camera")

        actual_height, actual_width = frame.shape[:2]
        if (actual_width, actual_height) != (self.spec.width, self.spec.height):
            capture.release()
            raise CameraError(
                f"{self.spec.name} camera returned {actual_width}x{actual_height}, "
                f"expected {self.spec.width}x{self.spec.height}"
            )
        self.capture = capture

    def read(self):
        if self.capture is None:
            raise CameraError(f"{self.spec.name} camera is not open")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise CameraError(f"Frame read failed for {self.spec.name} camera")
        if self.spec.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.spec.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.spec.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.spec.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.spec.flip_vertical:
            frame = cv2.flip(frame, 0)
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "UsbCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
