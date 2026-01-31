#!/usr/bin/env python3
"""High-level application for cone detection and tracking."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

import cv2

from .config import deep_merge, load_config, watch_config
from .detector import BaseDetector, ColorDetector, DetectorFusion
from .reasons_writer import ReasonsWriter
from .run_csv_logger import RunCSVLogger
from .tracker import MultiConeTracker
from .utils import ConeState
from .visualizer import Visualizer

logger = logging.getLogger(__name__)


class App:
    """Coordinate capture, detection, tracking, logging, and visualization."""

    def __init__(self, config_path: str = "cone_config.yaml") -> None:
        self.config_path = config_path
        self.config = load_config(self.config_path)
        self.detector = self._build_detector()
        self.tracker = MultiConeTracker(self.config)
        self.visualizer = Visualizer(self.config)
        self.csv_logger = RunCSVLogger()
        self.reasons_writer = ReasonsWriter()
        self.reasons_writer.set_start_timestamp()
        self.reasons_writer.set_config_summary(self.config)
        self._prev_confirmed_ids: set[int] = set()
        self._active_track_ids: set[int] = set()
        self._reload_msg: Optional[str] = None
        self._source_label = self.config["camera"].get("video_path") or f"cam{self.config['camera'].get('index', 0)}"

    def _build_detector(self) -> BaseDetector:
        det_cfg = self.config.get("detector", {}) or {}
        det_type = det_cfg.get("type", "color")
        params = det_cfg.get("params", {}) or {}
        if det_type == "fusion":
            strategies = params.get("strategies", [])
            detectors: List[BaseDetector] = []
            for strat in strategies:
                strat_type = strat.get("type") or "color"
                strat_params = strat.get("params", {})
                merged = deep_merge(self.config, strat_params)
                detectors.append(ColorDetector(merged))
            if detectors:
                return DetectorFusion(detectors)
        if det_type == "color":
            return ColorDetector(self.config)
        logger.warning("Unsupported detector type '%s', defaulting to color", det_type)
        return ColorDetector(self.config)

    def _open_capture(self) -> cv2.VideoCapture:
        cam = self.config["camera"]
        path = cam.get("video_path")
        backend = cam.get("backend")
        index_value = cam.get("index")
        index = int(index_value if index_value is not None else 0)
        if path:
            cap = cv2.VideoCapture(path)
        else:
            if backend:
                backend_enum = getattr(cv2, f"CAP_{backend.upper()}", cv2.CAP_ANY)
                cap = cv2.VideoCapture(index, backend_enum)
            else:
                cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open capture source '{path or index}'")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam.get("capture_width", 1920)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam.get("capture_height", 1080)))
        cap.set(cv2.CAP_PROP_FPS, int(cam.get("fps", 30)))
        return cap

    def _reload_config(self) -> None:
        self.config = load_config(self.config_path)
        self.detector = self._build_detector()
        self.tracker = MultiConeTracker(self.config)
        self.visualizer = Visualizer(self.config)
        self.reasons_writer.set_config_summary(self.config)
        self._reload_msg = f"Config reloaded {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
        self._source_label = self.config["camera"].get("video_path") or f"cam{self.config['camera'].get('index', 0)}"

    def run(self) -> None:
        try:
            cap = self._open_capture()
        except RuntimeError as exc:
            logger.error("Failed to start capture: %s", exc)
            return
        run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.csv_logger.open_if_enabled(self.config, self._source_label, run_ts)
        frame_idx = 0
        last_time = time.perf_counter()
        show_windows = bool(self.config["debug"].get("show_windows", True))
        show_mask = bool(self.config["debug"].get("show_mask", True))

        try:
            while True:
                if watch_config(self.config_path):
                    self._reload_config()
                    show_windows = bool(self.config["debug"].get("show_windows", True))
                    show_mask = bool(self.config["debug"].get("show_mask", True))

                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                now = time.perf_counter()
                dt = now - last_time if last_time else 1e-6
                fps = 1.0 / dt if dt > 0 else 0.0
                last_time = now

                process_width = int(self.config["camera"].get("process_width", frame.shape[1]))
                process_height = int(self.config["camera"].get("process_height", frame.shape[0]))
                if frame.shape[1] != process_width or frame.shape[0] != process_height:
                    frame_proc = cv2.resize(frame, (process_width, process_height))
                else:
                    frame_proc = frame

                results, mask, rejects = self.detector.detect(frame_proc)
                self.tracker.update(results)

                confirmed_ids = sorted(
                    t.track_id for t in self.tracker.tracks if t.state == ConeState.CONFIRMED
                )
                suspect_ids = sorted(
                    t.track_id for t in self.tracker.tracks if t.state == ConeState.SUSPECT
                )
                current_ids = {t.track_id for t in self.tracker.tracks}
                deleted_ids = sorted(self._active_track_ids - current_ids)
                new_confirmed = [tid for tid in confirmed_ids if tid not in self._prev_confirmed_ids]
                self._prev_confirmed_ids = set(confirmed_ids)
                self._active_track_ids = current_ids

                self.reasons_writer.add_frame_data(
                    frame_idx=frame_idx,
                    timestamp_ms=int(time.time() * 1000),
                    detections=results,
                    rejects=rejects,
                    tracker_events={"confirmed": new_confirmed, "deleted": deleted_ids},
                    track_states={"confirmed_ids": confirmed_ids, "suspect_ids": suspect_ids},
                )

                pos_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) if cap.get(cv2.CAP_PROP_POS_MSEC) >= 0 else 0
                self.csv_logger.log_frame(
                    frame_idx=frame_idx,
                    frame_w=frame_proc.shape[1],
                    tracks=self.tracker.tracks,
                    ts_wallclock_ms=int(time.time() * 1000),
                    ts_source_ms=pos_ms,
                    fps=fps,
                    hfov_deg=self.config["camera"].get("hfov_deg"),
                    cone_height_m=None,
                )

                drawn = self.visualizer.draw(frame_proc, self.tracker.tracks, rejects, fps, config_reload_msg=self._reload_msg)
                self._reload_msg = None

                if show_windows:
                    self.visualizer.show(drawn, mask, show_mask)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if show_windows:
                cv2.destroyAllWindows()
            self.csv_logger.close()
            if self.config["debug"].get("reasons_txt_enabled", False):
                self.reasons_writer.write_report(output_path=self.config["debug"].get("reasons_txt_path"))
