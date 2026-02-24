#!/usr/bin/env python3
"""Visualization for cone detection and tracking."""
import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .tracker import Track
from .utils import ConeState


# =========================
# VISUALIZER
# =========================
class Visualizer:
    """Visualize cone detection and tracking results."""
    
    MASK_NORMALIZED_EPS = 1e-3
    COLORS = [
        (0, 255, 0), (0, 200, 255), (255, 0, 0), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (180, 180, 0), (0, 180, 180),
    ]

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["debug"]
        self.camera_cfg = config["camera"]
        self.source_label = None

    def _color(self, track_id: int) -> Tuple[int, int, int]:
        """Get color for track ID."""
        return self.COLORS[track_id % len(self.COLORS)]

    @classmethod
    def _normalize_mask(cls, mask: np.ndarray) -> np.ndarray:
        if mask.dtype == np.bool_:
            return mask.astype(np.uint8) * 255
        if mask.dtype == np.uint8:
            return mask
        if np.issubdtype(mask.dtype, np.floating):
            if not np.any(mask > 0):
                return np.zeros(mask.shape, dtype=np.uint8)
            upper = float(np.max(mask))
            is_normalized = upper <= 1.0 + cls.MASK_NORMALIZED_EPS
            scale = 255.0 if is_normalized else 1.0  # scale normalized [0..1] masks to [0..255]
            return np.clip(mask * scale, 0, 255).astype(np.uint8)
        return mask.astype(np.uint8)
    
    def _compute_heading_info(self, track: Track, frame_w: int) -> Tuple[str, float]:
        """
        Compute heading direction and angle for a track.
        
        Args:
            track: Track object with cx (center x position)
            frame_w: Frame width in pixels
            
        Returns:
            Tuple of (direction_str, angle_deg) e.g. ("LEFT", -3.12)
        """
        # Get HFOV from config with fallback to 70 degrees
        hfov_deg = self.camera_cfg.get("hfov_deg", 70.0)
        
        # Validate HFOV is in a reasonable range (10-170 degrees)
        if hfov_deg < 10.0 or hfov_deg > 170.0:
            hfov_deg = 70.0
        
        hfov_rad = math.radians(hfov_deg)
        
        # Compute focal length using pinhole model: focal = (width/2) / tan(hfov/2)
        focal_px = (frame_w / 2.0) / math.tan(hfov_rad / 2.0)
        
        # Frame center
        center_x = frame_w / 2.0
        
        # Calculate horizontal error
        err_px = track.cx - center_x
        
        # Convert pixel error to angle using the horizontal field of view
        # angle = arctan(opposite/adjacent) = arctan(err_px / focal_px)
        angle_rad = math.atan(err_px / focal_px)
        angle_deg = math.degrees(angle_rad)
        
        # Get deadband threshold
        deadband_deg = self.cfg.get("heading_center_deadband_deg", 0.5)
        
        # Determine direction
        if abs(angle_deg) < deadband_deg:
            direction = "CENTER"
        elif angle_deg > 0:
            direction = "RIGHT"
        else:
            direction = "LEFT"
        
        return direction, angle_deg

    def draw(self, frame: np.ndarray, tracks: List[Track], rejects: List[Tuple[Tuple[int, int, int, int], str]], fps: float, config_reload_msg: str = None):
        """Draw tracking visualization on frame."""
        # Always draw annotations (for both video output and windows)
        # The show_windows flag only controls whether to display windows, not whether to draw
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.cfg.get("annotate_source_in_overlay", False) and self.source_label:
            cv2.putText(frame, f"SOURCE: {self.source_label}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Tracks info (improved with suspects)
        confirmed_count = sum(1 for t in tracks if t.state == ConeState.CONFIRMED)
        suspect_count = sum(1 for t in tracks if t.state == ConeState.SUSPECT)
        cv2.putText(frame, f"Tracks: {len(tracks)} ({confirmed_count} conf, {suspect_count} susp)", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add rejection counter
        cv2.putText(frame, f"Rejects: {len(rejects)}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Config reload message (adjusted Y position)
        if config_reload_msg:
            cv2.putText(frame, config_reload_msg, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # draw tracks
        for t in tracks:
            if t.state == ConeState.CONFIRMED:
                color = self._color(t.track_id)
            else:
                if not self.cfg.get("draw_suspects", False):
                    continue
                color = (0, 255, 255)

            x, y, w, h = t.bbox()
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # For CONFIRMED tracks, optionally show heading overlay
            if t.state == ConeState.CONFIRMED and self.cfg.get("show_heading_overlay", False):
                # Compute heading info
                direction, angle_deg = self._compute_heading_info(t, frame.shape[1])
                
                # Format heading text with sign
                heading_text = f"{direction} {angle_deg:+.2f}°"
                
                # Position heading text above the existing label
                # Use y-24 for heading line, y-10 for ID label
                # Clamp to avoid negative y positions
                heading_y = max(15, y - 24)
                label_y = max(30, y - 10)
                
                # Draw heading line (first line)
                cv2.putText(
                    frame,
                    heading_text,
                    (x, heading_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                )
                
                # Draw ID/state/avg label (second line)
                cv2.putText(
                    frame,
                    f"ID {t.track_id} {t.state.name} avg={t.avg_score():.2f}",
                    (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                )
            else:
                # Original label without heading overlay
                cv2.putText(
                    frame,
                    f"ID {t.track_id} {t.state.name} avg={t.avg_score():.2f}",
                    (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                )

        # rejections - improved visibility
        if self.cfg.get("show_rejection_reason", False):
            for bbox, reason in rejects:
                x, y, w, h = bbox
                # More visible rectangle (thickness 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Larger, more readable text
                cv2.putText(frame, reason, (x, max(0, y - 5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    def show(self, frame: np.ndarray, mask: np.ndarray, show_mask: bool) -> None:
        cv2.imshow("Tracker", frame)
        if show_mask and mask is not None:
            cv2.imshow("Mask", self._normalize_mask(mask))
