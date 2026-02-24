#!/usr/bin/env python3
"""
Utility module for writing detailed reasons reports (.txt files).

This module provides functionality to collect and write per-frame detection/tracking
reasons (accepted, rejected, suspect) to a human-readable text file.
"""
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import log_event

logger = logging.getLogger(__name__)


class ReasonsWriter:
    """
    Collects per-frame detection and tracking reasons and writes them to a .txt report.
    
    This class accumulates frame-by-frame information about:
    - Accepted detections (bbox, score, geometric data)
    - Rejected candidates (bbox, rejection reason)
    - Tracker events (confirmations, deletions)
    - Track states (confirmed, suspect)
    
    At the end of a run, it writes a formatted .txt report with:
    - Header: timestamp, configuration summary
    - Per-frame sections: detections, rejections, tracker events
    - Footer: summary statistics
    """
    
    def __init__(self):
        self.frames_data: List[Dict[str, Any]] = []
        self.start_timestamp: Optional[str] = None
        self.config_summary: Dict[str, Any] = {}
        self.output_path: Optional[str] = None
        
    def set_start_timestamp(self, timestamp: Optional[str] = None):
        """Set the run start timestamp (ISO format or auto-generate).

        Units: ISO 8601 string.
        Complexity: O(1).
        """
        if timestamp:
            self.start_timestamp = timestamp
        else:
            self.start_timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    
    def set_config_summary(self, config: Dict[str, Any]):
        """Extract relevant config for header summary.

        Units: values are raw config values.
        Complexity: O(1).
        """
        self.config_summary = {
            "hsv_orange_low_1": config.get("hsv_orange", {}).get("low_1", []),
            "hsv_orange_high_1": config.get("hsv_orange", {}).get("high_1", []),
            "tracking_lost_timeout": config.get("tracking", {}).get("lost_timeout", 0.6),
            "min_frames_for_confirm": config.get("tracking", {}).get("min_frames_for_confirm", 6),
            "structured_logging": config.get("debug", {}).get("structured_logging", False),
        }
    
    def add_frame_data(
        self,
        frame_idx: int,
        timestamp_ms: Optional[int] = None,
        detections: Optional[List[Tuple[Tuple[int, int, int, int], float, Dict[str, Any]]]] = None,
        rejects: Optional[List[Tuple[Tuple[int, int, int, int], str]]] = None,
        gist_candidates: Optional[List[Dict[str, Any]]] = None,
        tracker_events: Optional[Dict[str, Any]] = None,
        track_states: Optional[Dict[str, List[int]]] = None,
    ):
        """
        Add data for a single frame.
        
        Args:
            frame_idx: Frame index
            timestamp_ms: Optional timestamp in milliseconds
            detections: List of (bbox, score, data) for accepted detections
            rejects: List of (bbox, reason) for rejected candidates
            gist_candidates: Optional list of gist pipeline candidates
            tracker_events: Dict with 'confirmed' and 'deleted' lists of track IDs
            track_states: Dict with 'confirmed_ids' and 'suspect_ids' lists

        Units:
            - frame_idx: frames
            - timestamp_ms: ms
            - bbox: px
            - score: normalized 0..1
        Complexity: O(1) per call (data append only).
        """
        frame_data = {
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms or 0,
            "detections": detections or [],
            "rejects": rejects or [],
            "gist_candidates": gist_candidates or [],
            "tracker_events": tracker_events or {"confirmed": [], "deleted": []},
            "track_states": track_states or {"confirmed_ids": [], "suspect_ids": []},
        }
        self.frames_data.append(frame_data)
    
    def write_report(
        self,
        output_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        iou_hist_path: Optional[str] = None,
    ) -> str:
        """
        Write the complete reasons report to a .txt file.
        
        Args:
            output_path: Path to output file (if None, auto-generate in repo root)
            csv_path: Optional path to CSV file (for footer reference)
            iou_hist_path: Optional path to IoU histogram CSV (for footer reference)
            
        Returns:
            Path to the written file
            
        Raises:
            IOError: If file cannot be written

        Units: paths are strings.
        Complexity: O(n) in number of frames and logged items.
        """
        # Determine output path
        if output_path:
            self.output_path = output_path
        else:
            # Auto-generate path in repo root
            safe_ts = self.start_timestamp.replace(":", "-").replace(".", "-") if self.start_timestamp else "unknown"
            self.output_path = f"./reasons_{safe_ts}.txt"
        
        # Ensure parent directory exists
        parent_dir = os.path.dirname(self.output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                # Write header
                f.write("=" * 80 + "\n")
                f.write("CONE DETECTION REASONS REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Start Timestamp: {self.start_timestamp}\n")
                f.write(f"Total Frames: {len(self.frames_data)}\n")
                f.write("\n")
                
                # Write config summary
                f.write("Configuration Summary:\n")
                f.write(f"  HSV Orange Range 1: {self.config_summary.get('hsv_orange_low_1', [])} - {self.config_summary.get('hsv_orange_high_1', [])}\n")
                f.write(f"  Tracking Lost Timeout: {self.config_summary.get('tracking_lost_timeout', 0.6)}s\n")
                f.write(f"  Min Frames for Confirm: {self.config_summary.get('min_frames_for_confirm', 6)}\n")
                f.write("\n")
                f.write("=" * 80 + "\n\n")
                
                # Write per-frame data
                total_accepted = 0
                total_rejected = 0
                total_confirmed = 0
                
                for frame_data in self.frames_data:
                    frame_idx = frame_data["frame_idx"]
                    timestamp_ms = frame_data["timestamp_ms"]
                    detections = frame_data["detections"]
                    rejects = frame_data["rejects"]
                    gist_candidates = frame_data["gist_candidates"]
                    tracker_events = frame_data["tracker_events"]
                    track_states = frame_data["track_states"]
                    
                    f.write(f"Frame {frame_idx} (ts={timestamp_ms}ms)\n")
                    f.write("-" * 80 + "\n")
                    
                    # Detector results
                    f.write(f"  Detector:\n")
                    if detections:
                        f.write(f"    ACCEPTED ({len(detections)}):\n")
                        for bbox, score, data in detections:
                            f.write(f"      - bbox={bbox}, score={score:.3f}\n")
                            f.write(f"        data: profile={data.get('profile', 0):.3f}, fill={data.get('fill', 0):.3f}, ")
                            f.write(f"aspect={data.get('aspect', 0):.2f}, parts={data.get('parts', 0)}\n")
                            total_accepted += 1
                    else:
                        f.write(f"    ACCEPTED (0)\n")
                    
                    if rejects:
                        f.write(f"    REJECTED ({len(rejects)}):\n")
                        for bbox, reason in rejects:
                            f.write(f"      - bbox={bbox}, reason: {reason}\n")
                            total_rejected += 1
                    else:
                        f.write(f"    REJECTED (0)\n")
                    
                    # Gist candidates (if available)
                    if gist_candidates:
                        f.write(f"  Gist:\n")
                        f.write(f"    CANDIDATES ({len(gist_candidates)}):\n")
                        for candidate in gist_candidates:
                            bbox = candidate.get("bbox", (0, 0, 0, 0))
                            score = candidate.get("score", 0.0)
                            reason = candidate.get("reason", "geometric validation")
                            f.write(f"      - bbox={bbox}, score={score:.3f}, reason: {reason}\n")
                    
                    # Tracker events
                    confirmed_ids = tracker_events.get("confirmed", [])
                    deleted_ids = tracker_events.get("deleted", [])
                    
                    if confirmed_ids or deleted_ids:
                        f.write(f"  Tracker:\n")
                        if confirmed_ids:
                            f.write(f"    CONFIRMED: {confirmed_ids}\n")
                            total_confirmed += len(confirmed_ids)
                        if deleted_ids:
                            f.write(f"    DELETED: {deleted_ids}\n")
                    
                    # Track states
                    confirmed_state_ids = track_states.get("confirmed_ids", [])
                    suspect_state_ids = track_states.get("suspect_ids", [])
                    
                    if confirmed_state_ids or suspect_state_ids:
                        f.write(f"  Track States:\n")
                        if confirmed_state_ids:
                            f.write(f"    CONFIRMED: {confirmed_state_ids}\n")
                        if suspect_state_ids:
                            f.write(f"    SUSPECT: {suspect_state_ids}\n")
                    
                    f.write("\n")
                
                # Write footer
                f.write("=" * 80 + "\n")
                f.write("SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total Frames: {len(self.frames_data)}\n")
                f.write(f"Total Accepted Detections: {total_accepted}\n")
                f.write(f"Total Rejected Detections: {total_rejected}\n")
                f.write(f"Total Confirmed Tracks: {total_confirmed}\n")
                if csv_path:
                    f.write(f"CSV Output: {csv_path}\n")
                if iou_hist_path:
                    f.write(f"IoU Histogram CSV: {iou_hist_path}\n")
                f.write(f"Reasons Report: {self.output_path}\n")
                f.write("=" * 80 + "\n")
            
            log_event(
                logger,
                {"debug": {"structured_logging": bool(self.config_summary.get("structured_logging", False))}},
                logging.INFO,
                f"Reasons report written to: {self.output_path}",
                "reasons_report_written",
                path=self.output_path,
            )
            return self.output_path
            
        except IOError as e:
            log_event(
                logger,
                {"debug": {"structured_logging": bool(self.config_summary.get("structured_logging", False))}},
                logging.ERROR,
                f"Failed to write reasons report to {self.output_path}: {e}",
                "reasons_report_failed",
                path=self.output_path,
                error=str(e),
            )
            raise
