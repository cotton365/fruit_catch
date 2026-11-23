from __future__ import annotations

import unittest

import cv2
import numpy as np

from vision.detectors import SideColorDetector, TopCenterDetector
from vision.protocol import StartSignalParser, alignment_message, orientation_message, result_message


class DetectorTests(unittest.TestCase):
    def test_top_center_offset(self) -> None:
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        green = (0, 190, 0)
        cv2.line(frame, (350, 300), (350, 190), green, 25)
        cv2.line(frame, (350, 200), (270, 110), green, 25)
        cv2.line(frame, (350, 200), (430, 110), green, 25)
        config = {
            "green_hsv_lower": [35, 45, 30],
            "green_hsv_upper": [95, 255, 255],
            "roi": [0.0, 0.0, 1.0, 1.0],
            "min_component_area_px": 1000,
            "open_kernel_px": 3,
            "close_kernel_px": 5,
            "reference_offset_px": [0, 0],
        }
        detection, _ = TopCenterDetector(config).detect(frame)
        self.assertTrue(detection.found)
        self.assertIsNotNone(detection.dx_px)
        self.assertGreater(detection.dx_px, 35)
        self.assertIsNotNone(detection.orientation_deg)

    def test_side_detects_green_but_not_yellow(self) -> None:
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        green = (0, 190, 0)
        yellow = (0, 220, 220)
        cv2.ellipse(frame, (90, 100), (24, 34), 0, 0, 360, green, -1)
        cv2.ellipse(frame, (180, 300), (24, 34), 0, 0, 360, yellow, -1)
        cv2.ellipse(frame, (510, 100), (24, 34), 0, 0, 360, yellow, -1)
        cv2.ellipse(frame, (420, 300), (24, 34), 0, 0, 360, green, -1)
        config = {
            "green_hsv_lower": [35, 55, 30],
            "green_hsv_upper": [95, 255, 255],
            "open_kernel_px": 3,
            "close_kernel_px": 3,
            "min_blob_area_px": 300,
            "max_blob_area_px": 10000,
            "min_fill_ratio": 0.3,
            "max_elongation": 2.2,
            "min_circularity": 0.15,
            "positions": {
                "L1": [0.0, 0.0, 0.3, 0.45],
                "L2": [0.0, 0.55, 0.45, 1.0],
                "R1": [0.7, 0.0, 1.0, 0.45],
                "R2": [0.55, 0.55, 1.0, 1.0],
            },
        }
        detection, _ = SideColorDetector(config).detect(frame)
        self.assertEqual(detection.green_positions, ("L1", "R2"))

    def test_protocol(self) -> None:
        self.assertTrue(StartSignalParser.has_start(b"S"))
        self.assertTrue(StartSignalParser.has_start(b"s\r\n"))
        self.assertFalse(StartSignalParser.has_start(b"\r\n"))
        self.assertEqual(alignment_message(3, -12, 7), "A,3,-12,+7")
        self.assertEqual(orientation_message(3, -12.34), "T,3,-12.3")
        self.assertEqual(result_message(3, ["R2", "L1"]), "G,3,L1,R2")
        self.assertEqual(result_message(3, []), "G,3,NONE")


if __name__ == "__main__":
    unittest.main()
