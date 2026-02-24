#!/usr/bin/env python3
"""Utility functions for cone detection."""
import json
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =========================
# STATE / UTILS
# =========================
class ConeState(Enum):
    """Enum for track lifecycle states."""
    SUSPECT = auto()
    CONFIRMED = auto()
    LOST = auto()


class VisibilityState(Enum):
    """Enum for per-frame visibility state."""
    VISIBLE = auto()
    OCCLUDED = auto()


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp value v between lo and hi.

    Units: same as input value.
    Complexity: O(1).
    """
    return max(lo, min(hi, v))


def safe_roi(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract a safe ROI from image given a bounding box.

    Units: px for bbox coordinates.
    Complexity: O(1) for slicing references; no copy if possible.
    """
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x = int(clamp(x, 0, w - 1))
    y = int(clamp(y, 0, h - 1))
    x2 = int(clamp(x + bw, 0, w))
    y2 = int(clamp(y + bh, 0, h))
    if x2 <= x or y2 <= y:
        return img[0:0, 0:0], (0, 0, 0, 0)
    return img[y:y2, x:x2], (x, y, x2 - x, y2 - y)


def x_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate horizontal overlap ratio between two bounding boxes.

    Units: px for bbox coordinates; returns normalized ratio in [0,1].
    Complexity: O(1).
    """
    ax, _, aw, _ = a
    bx, _, bw, _ = b
    if aw <= 0 or bw <= 0:
        return 0.0
    a1, a2 = ax, ax + aw
    b1, b2 = bx, bx + bw
    inter = max(0, min(a2, b2) - max(a1, b1))
    denom = float(min(aw, bw))
    if denom <= 0:
        return 0.0
    return inter / denom


def bbox_union(boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    """Calculate the union bounding box of a list of boxes.

    Units: px for bbox coordinates.
    Complexity: O(n).
    """
    if not boxes:
        return None
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2 = [b[0] + b[2] for b in boxes]
    y2 = [b[1] + b[3] for b in boxes]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(x2)
    y_max = max(y2)
    w = max(0, x_max - x_min)
    h = max(0, y_max - y_min)
    return (x_min, y_min, w, h)


def bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate the center point of a bounding box.

    Units: px for bbox coordinates; returns center in px.
    Complexity: O(1).
    """
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def bbox_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate Euclidean distance between centers of two bounding boxes.

    Units: px for bbox coordinates; returns distance in px.
    Complexity: O(1).
    """
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(np.hypot(ax - bx, ay - by))


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute intersection-over-union (IoU) of two bounding boxes.

    Units: px for bbox coordinates; returns normalized ratio in [0,1].
    Complexity: O(1).
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    union = float(aw * ah + bw * bh - inter)
    if union <= 0:
        return 0.0
    return inter / union


def log_event(logger: logging.Logger, config: Dict[str, Any], level: int, message: str, event_type: str, **fields: Any) -> None:
    """Log either structured JSON or plain text depending on config.

    Units: free-form fields depend on event type.
    Complexity: O(1).
    Errors: Never raises; falls back to plain text on JSON issues.
    """
    structured = bool(config.get("debug", {}).get("structured_logging", False))
    if not structured:
        logger.log(level, message)
        return
    payload = {"event": event_type, **fields}
    try:
        logger.log(level, json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.log(level, message)
