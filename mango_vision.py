#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision.camera import CameraError, CameraSpec, UsbCamera
from vision.detectors import SideColorDetector, TopCenterDetector
from vision.protocol import (
    SerialLink,
    alignment_message,
    begin_message,
    centered_message,
    error_message,
    orientation_message,
    result_message,
)
from vision.preview import PreviewServer


LOGGER = logging.getLogger("mango_vision")


class VisionController:
    def __init__(
        self,
        config: dict[str, Any],
        serial_link: SerialLink | None,
        start_now: bool,
        once: bool,
        debug_dir: Path | None,
        preview: PreviewServer | None,
    ):
        self.config = config
        self.serial_link = serial_link
        self.once = once
        self.debug_dir = debug_dir
        self.preview = preview
        self.running = True
        self.sequence = 0
        self.state = "IDLE"
        self.camera: UsbCamera | None = None
        self.camera_warmup_remaining = 0
        self.failed_reads = 0
        self.last_error_time = 0.0
        self.last_alignment_tx = 0.0
        self.stable_count = 0
        self.dx_history: deque[float] = deque()
        self.dy_history: deque[float] = deque()
        self.angle_history: deque[float] = deque()
        self.side_votes: dict[str, int] = {name: 0 for name in SideColorDetector.POSITION_ORDER}
        self.side_frame_count = 0
        self.last_top_debug = None
        self.last_side_debug = None

        self.top_spec = CameraSpec.from_config("top", config["cameras"]["top"])
        self.side_spec = CameraSpec.from_config("side", config["cameras"]["side"])
        self.top_detector = TopCenterDetector(config["top_detector"])
        self.side_detector = SideColorDetector(config["side_detector"])

        window = int(config["top_detector"].get("smoothing_window", 5))
        self.dx_history = deque(maxlen=max(1, window))
        self.dy_history = deque(maxlen=max(1, window))
        self.angle_history = deque(maxlen=max(1, window))

        if start_now:
            self.start_cycle()

    def send(self, message: str) -> None:
        LOGGER.info("TX %s", message)
        if self.serial_link is not None:
            self.serial_link.send(message)

    def stop(self, *_args) -> None:
        self.running = False

    def close_camera(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None

    def open_camera(self, spec: CameraSpec) -> None:
        self.close_camera()
        camera = UsbCamera(spec)
        camera.open()
        self.camera = camera
        self.camera_warmup_remaining = 0
        self.failed_reads = 0
        LOGGER.info("Opened %s camera: %s", spec.name, spec.device)

    def start_cycle(self) -> None:
        self.sequence += 1
        self.state = "ALIGN"
        self.stable_count = 0
        self.dx_history.clear()
        self.dy_history.clear()
        self.angle_history.clear()
        self.side_votes = {name: 0 for name in SideColorDetector.POSITION_ORDER}
        self.side_frame_count = 0
        self.last_top_debug = None
        self.last_side_debug = None
        self.close_camera()
        self.send(begin_message(self.sequence))
        LOGGER.info("Cycle %d started", self.sequence)

    def report_error_limited(self, code: str) -> None:
        now = time.monotonic()
        if now - self.last_error_time >= 1.0:
            self.last_error_time = now
            self.send(error_message(self.sequence, code))

    def ensure_camera(self, spec: CameraSpec) -> bool:
        if self.camera is not None and self.camera.spec == spec:
            return True
        try:
            self.open_camera(spec)
            return True
        except CameraError as exc:
            LOGGER.error("%s", exc)
            self.report_error_limited(f"{spec.name.upper()}_CAMERA")
            self.close_camera()
            time.sleep(0.2)
            return False

    def read_frame(self):
        assert self.camera is not None
        try:
            frame = self.camera.read()
            self.failed_reads = 0
            if self.camera_warmup_remaining > 0:
                self.camera_warmup_remaining -= 1
                return None
            return frame
        except CameraError as exc:
            self.failed_reads += 1
            LOGGER.warning("%s", exc)
            if self.failed_reads >= 5:
                self.report_error_limited(f"{self.camera.spec.name.upper()}_READ")
                self.close_camera()
            return None

    def process_alignment(self) -> None:
        if not self.ensure_camera(self.top_spec):
            return
        frame = self.read_frame()
        if frame is None:
            return

        detection, annotated = self.top_detector.detect(frame)
        self.last_top_debug = annotated
        self.publish_preview(annotated, "top")
        if not detection.found:
            self.stable_count = 0
            self.dx_history.clear()
            self.dy_history.clear()
            self.report_error_limited("TREE_NOT_FOUND")
            return

        assert detection.dx_px is not None and detection.dy_px is not None
        self.dx_history.append(detection.dx_px)
        self.dy_history.append(detection.dy_px)
        if detection.orientation_deg is not None:
            self.angle_history.append(detection.orientation_deg)
        dx = int(round(float(np.median(self.dx_history))))
        dy = int(round(float(np.median(self.dy_history))))

        now = time.monotonic()
        period = float(self.config["top_detector"].get("transmit_period_s", 0.15))
        if now - self.last_alignment_tx >= period:
            self.last_alignment_tx = now
            self.send(alignment_message(self.sequence, dx, dy))

        tolerance_x, tolerance_y = self.config["top_detector"].get("tolerance_px", [15, 15])
        if abs(dx) <= int(tolerance_x) and abs(dy) <= int(tolerance_y):
            self.stable_count += 1
        else:
            self.stable_count = 0

        required = int(self.config["top_detector"].get("stable_frames", 5))
        if self.stable_count >= required and self.angle_history:
            angle_deg = float(np.median(self.angle_history)) + float(
                self.config["top_detector"].get("orientation_offset_deg", 0.0)
            )
            while angle_deg >= 90.0:
                angle_deg -= 180.0
            while angle_deg < -90.0:
                angle_deg += 180.0
            self.send(orientation_message(self.sequence, angle_deg))
            # C is the final barrier: the MCU may enter the next stage only after
            # it has already received the orientation angle.
            self.send(centered_message(self.sequence))
            self.save_debug("top_centered", self.last_top_debug)
            self.state = "SIDE"
            self.close_camera()
            LOGGER.info("Alignment complete for cycle %d", self.sequence)

    def process_side(self) -> None:
        if not self.ensure_camera(self.side_spec):
            return
        frame = self.read_frame()
        if frame is None:
            return

        detection, annotated = self.side_detector.detect(frame)
        self.last_side_debug = annotated
        self.publish_preview(annotated, "side")
        self.side_frame_count += 1
        for position in detection.green_positions:
            self.side_votes[position] += 1

        sample_frames = int(self.config["side_detector"].get("sample_frames", 15))
        if self.side_frame_count < sample_frames:
            return

        required = int(self.config["side_detector"].get("required_positive_frames", 9))
        green_positions = [
            position
            for position in SideColorDetector.POSITION_ORDER
            if self.side_votes[position] >= required
        ]
        self.send(result_message(self.sequence, green_positions))
        LOGGER.info("Side votes: %s", self.side_votes)
        self.save_debug("side_result", self.last_side_debug)
        self.close_camera()
        self.state = "DONE"
        if self.once:
            self.running = False

    def process_idle_preview(self) -> None:
        if self.preview is None:
            time.sleep(0.02)
            return
        if not self.ensure_camera(self.top_spec):
            return
        frame = self.read_frame()
        if frame is None:
            return
        _, annotated = self.top_detector.detect(frame)
        self.publish_preview(annotated, "top")

    def publish_preview(self, frame, camera: str) -> None:
        if self.preview is not None:
            self.preview.publish(frame, state=self.state, camera=camera, sequence=self.sequence)

    def save_debug(self, label: str, frame) -> None:
        if self.debug_dir is None or frame is None:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self.debug_dir / f"cycle_{self.sequence:03d}_{label}.jpg"
        cv2.imwrite(str(path), frame)
        LOGGER.info("Saved %s", path)

    def run(self) -> int:
        LOGGER.info("Waiting for S/s on the serial port")
        try:
            while self.running:
                if self.serial_link is not None and self.serial_link.poll_start():
                    self.start_cycle()

                if self.state == "ALIGN":
                    self.process_alignment()
                elif self.state == "SIDE":
                    self.process_side()
                elif self.state == "IDLE":
                    self.process_idle_preview()
                else:
                    time.sleep(0.02)
        finally:
            self.close_camera()
            if self.serial_link is not None:
                self.serial_link.close()
            if self.preview is not None:
                self.preview.stop()
        return 0


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDK X5 mango thinning vision controller")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--no-serial", action="store_true", help="Print protocol messages instead of opening UART")
    parser.add_argument("--start-now", action="store_true", help="Start one cycle immediately without waiting for S/s")
    parser.add_argument("--once", action="store_true", help="Exit after one completed cycle")
    parser.add_argument("--debug-dir", type=Path, help="Save annotated result images to this directory")
    parser.add_argument("--preview-port", type=int, default=8080, help="Web preview port (default: 8080)")
    parser.add_argument("--no-preview", action="store_true", help="Disable the web camera preview")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.start_now and not args.no_serial:
        LOGGER.warning("--start-now bypasses the MCU trigger")

    config = load_config(args.config)
    serial_link = None
    if not args.no_serial:
        serial_config = config["serial"]
        try:
            serial_link = SerialLink(
                port=str(serial_config["port"]),
                baudrate=int(serial_config.get("baudrate", 115200)),
                timeout_s=float(serial_config.get("timeout_s", 0.0)),
            )
        except Exception as exc:
            LOGGER.error("Cannot open serial port: %s", exc)
            return 2

    preview = None
    if not args.no_preview:
        preview = PreviewServer(args.preview_port)
        try:
            preview.start()
        except OSError as exc:
            LOGGER.error("Cannot start preview server on port %d: %s", args.preview_port, exc)
            if serial_link is not None:
                serial_link.close()
            return 3

    controller = VisionController(
        config=config,
        serial_link=serial_link,
        start_now=args.start_now,
        once=args.once,
        debug_dir=args.debug_dir,
        preview=preview,
    )
    signal.signal(signal.SIGINT, controller.stop)
    signal.signal(signal.SIGTERM, controller.stop)
    return controller.run()


if __name__ == "__main__":
    sys.exit(main())
