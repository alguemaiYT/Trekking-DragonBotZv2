#!/usr/bin/env python3
"""Color processing utilities for cone detection."""
import logging
import os
from typing import Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =========================
# COLOR HELPERS
# =========================
def gray_world(bgr: np.ndarray) -> np.ndarray:
    """Simple gray-world color normalization."""
    img = bgr.astype(np.float32)
    avg = img.mean(axis=(0, 1))
    avg_gray = float(np.mean(avg))
    # avoid division by zero
    scale = avg_gray / (avg + 1e-6)
    img = img * scale.reshape((1, 1, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def rg_chromaticity_mask(bgr: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """Compute simple r,g chromaticity mask. thresholds values in [0..1]."""
    img = bgr.astype(np.float32)
    s = img.sum(axis=2) + 1e-6
    r = img[:, :, 2] / s
    g = img[:, :, 1] / s
    cond = (r >= thresholds["r_min"]) & (r <= thresholds["r_max"]) & (g >= thresholds["g_min"]) & (g <= thresholds["g_max"])
    return (cond.astype(np.uint8) * 255)


def load_backproj_hist(path: str):
    """Try to load a histogram saved as numpy .npy (HxW)."""
    if not path or not os.path.exists(path):
        return None
    try:
        hist = np.load(path)
        # ensure float32
        hist = hist.astype(np.float32)
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist
    except Exception as e:
        logger.warning(f"Failed to load backproj histogram '{path}': {e}")
        return None


def mask_from_backproj(hsv: np.ndarray, hist: np.ndarray, thresh: int = 50) -> np.ndarray:
    """Given an HxS hist normalized 0..255, compute backprojection mask."""
    if hist is None:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    back = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    _, mask = cv2.threshold(back, thresh, 255, cv2.THRESH_BINARY)
    return mask
