#!/usr/bin/env python3
"""Regression test for detector output using a synthetic golden frame."""
import sys

import cv2
import numpy as np

from cone_tracker.config import DEFAULT_CONFIG
from cone_tracker.detector import ConeDetector


def _make_golden_frame(width=960, height=540):
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    # Orange-like HSV within configured range
    hsv_color = np.array([15, 200, 200], dtype=np.uint8)
    x, y, w, h = 200, 150, 80, 160
    hsv[y : y + h, x : x + w] = hsv_color
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr, (x, y, w, h)


def _mean_abs_diff(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b)) / 4.0


def main() -> int:
    cfg = DEFAULT_CONFIG.copy()
    cfg["geometry"] = dict(cfg["geometry"])
    cfg["geometry"]["min_group_area"] = 200
    cfg["geometry"]["min_frame_score"] = 0.1
    cfg["geometry"]["min_fill_ratio"] = 0.02
    cfg["geometry"]["max_fill_ratio"] = 0.98

    detector = ConeDetector(cfg)
    frame, expected_bbox = _make_golden_frame()
    results, _mask, _rejects = detector.detect(frame)
    if not results:
        print("No detections found.")
        return 1

    detected_bbox = results[0][0]
    diff = _mean_abs_diff(detected_bbox, expected_bbox)
    if diff > 2.0:
        print(f"BBox mismatch: detected={detected_bbox}, expected={expected_bbox}, mean_diff={diff:.2f}")
        return 1
    print("Regression test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
