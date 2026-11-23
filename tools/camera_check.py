#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.camera import CameraSpec, UsbCamera
from vision.detectors import SideColorDetector, TopCenterDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture and annotate both configured cameras")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "camera_check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    for name in ("top", "side"):
        spec = CameraSpec.from_config(name, config["cameras"][name])
        with UsbCamera(spec) as camera:
            frame = camera.read()
        assert frame is not None
        cv2.imwrite(str(args.output / f"{name}_raw.jpg"), frame)
        if name == "top":
            detection, annotated = TopCenterDetector(config["top_detector"]).detect(frame)
            print(f"top: found={detection.found} dx={detection.dx_px} dy={detection.dy_px}")
        else:
            detection, annotated = SideColorDetector(config["side_detector"]).detect(frame)
            print(f"side: green_positions={','.join(detection.green_positions) or 'NONE'}")
        cv2.imwrite(str(args.output / f"{name}_annotated.jpg"), annotated)

    print(f"Images saved in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
