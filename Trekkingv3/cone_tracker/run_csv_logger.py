#!/usr/bin/env python3
"""Async CSV logger for debug runs."""
import logging
import math
import os
import queue
import re
import threading
import uuid
from typing import Iterable, Optional, Sequence, TYPE_CHECKING

from .utils import clamp, log_event

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .tracker import Track

CSV_HEADER = (
    "frame_idx",
    "ts_wallclock_ms",
    "ts_source_ms",
    "source",
    "detected",
    "target_id",
    "cx",
    "cy",
    "err_px",
    "err_norm",
    "err_deg",
    "bbox_h",
    "est_dist_m",
    "avg_score",
    "fps",
)

DEFAULT_CSV_FILENAME_PATTERN = "run_{source}_{start_ts}.csv"
MAX_FILENAME_ATTEMPTS = 1000
UUID_FILENAME_ATTEMPTS = 10
DEFAULT_FLUSH_EVERY_FRAMES = 10
MIN_BBOX_HEIGHT_FOR_DISTANCE = 10.0
HFOV_MIN_DEG = 10.0  # Keep in sync with app HFOV validation bounds
HFOV_MAX_DEG = 170.0  # Keep in sync with app HFOV validation bounds

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_timestamp(timestamp: str) -> str:
    """Sanitize timestamp string for filesystem-safe usage.

    Units: string.
    Complexity: O(n) over string length.
    """
    return (timestamp or "").replace(":", "-").replace(".", "-")


def _sanitize_filename(name: str, source_label: str, safe_ts: str) -> str:
    """Generate a safe CSV filename.

    Units: string.
    Complexity: O(n).
    """
    name = os.path.basename(name or "")
    name = _FILENAME_SAFE_RE.sub("_", name).strip("._")
    if not name:
        name = f"run_{source_label}_{safe_ts}".strip("_")
    base, ext = os.path.splitext(name)
    if not base:
        base = "run"
    if ext.lower() != ".csv":
        ext = ".csv"
    return f"{base}{ext}"


def _focal_px(frame_w: int, hfov_deg: Optional[float]) -> Optional[float]:
    """Compute focal length in px from horizontal FoV.

    Units: frame_w in px, hfov_deg in degrees; returns focal length in px.
    Complexity: O(1).
    """
    if frame_w <= 0 or hfov_deg is None:
        return None
    if hfov_deg < HFOV_MIN_DEG or hfov_deg > HFOV_MAX_DEG:
        return None
    hfov_rad = math.radians(hfov_deg)
    return (frame_w / 2.0) / math.tan(hfov_rad / 2.0)


class RunCSVLogger:
    """Non-blocking CSV logger for per-frame debug metrics."""

    def __init__(self) -> None:
        self._enabled = False
        self.csv_path: Optional[str] = None
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._handle = None
        self._flush_every_frames = DEFAULT_FLUSH_EVERY_FRAMES
        self._source_label = "unknown"
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Return whether the CSV logger is enabled.

        Complexity: O(1).
        """
        return self._enabled

    def open_if_enabled(self, config: dict, source_label: str, run_start_ts: str) -> Optional[str]:
        """Open CSV file and start worker thread if enabled in config.

        Units: run_start_ts in ISO string; returns path string.
        Complexity: O(1) plus filesystem operations.
        Errors: returns None on failure, logs warnings/errors.
        """
        debug_cfg = (config or {}).get("debug", {})
        csv_cfg = debug_cfg.get("csv_export", {})
        enabled = False
        if isinstance(csv_cfg, dict):
            enabled = bool(csv_cfg.get("enabled", False))
        elif isinstance(csv_cfg, bool):
            enabled = csv_cfg
            csv_cfg = {}
        if not enabled:
            return None

        with self._lock:
            if self._enabled:
                return self.csv_path
            self._source_label = source_label or "unknown"
            output_dir = csv_cfg.get("csv_dir") or csv_cfg.get("output_dir", "logs")
            flush_setting = csv_cfg.get("flush_every_frames", DEFAULT_FLUSH_EVERY_FRAMES)
            try:
                self._flush_every_frames = max(1, int(flush_setting))
            except (TypeError, ValueError):
                log_event(
                    logger,
                    config or {},
                    logging.WARNING,
                    f"⚠️ Invalid flush_every_frames={flush_setting}, using default {DEFAULT_FLUSH_EVERY_FRAMES}",
                    "csv_invalid_flush",
                    flush_setting=flush_setting,
                    default=DEFAULT_FLUSH_EVERY_FRAMES,
                )
                self._flush_every_frames = DEFAULT_FLUSH_EVERY_FRAMES
            self._prepare_directory(output_dir)
            safe_ts = _sanitize_timestamp(run_start_ts)
            filename_pattern = csv_cfg.get("filename_pattern", DEFAULT_CSV_FILENAME_PATTERN)
            filename = _sanitize_filename(filename_pattern.format(source=self._source_label, start_ts=safe_ts), self._source_label, safe_ts)
            candidate = self._unique_filename(output_dir, filename)
            self.csv_path = os.path.join(output_dir, candidate)
            include_header = csv_cfg.get("include_header", True)
            try:
                self._handle = open(self.csv_path, "w", encoding="utf-8", newline="")
                if include_header:
                    self._handle.write(",".join(CSV_HEADER) + "\n")
                self._handle.flush()
            except OSError as exc:
                self._handle = None
                self.csv_path = None
                log_event(
                    logger,
                    config or {},
                    logging.ERROR,
                    f"⚠️ Failed to open CSV log: {exc}",
                    "csv_open_failed",
                    error=str(exc),
                )
                return None

            self._queue = queue.Queue()
            self._enabled = True
            self._worker = threading.Thread(
                target=self._worker_loop,
                args=(self._queue,),
                name="RunCSVLogger",
                daemon=True,
            )
            self._worker.start()
            log_event(
                logger,
                config or {},
                logging.INFO,
                f"📝 CSV export enabled: {self.csv_path}",
                "csv_enabled",
                path=self.csv_path,
            )
            return self.csv_path

    def log_frame(
        self,
        frame_idx: int,
        frame_w: int,
        tracks: Optional[Sequence["Track"]],
        ts_wallclock_ms: int,
        ts_source_ms: int,
        fps: float,
        hfov_deg: Optional[float],
        cone_height_m: Optional[float],
    ) -> None:
        """Queue a frame log entry for CSV output.

        Units:
            - frame_idx: frames
            - frame_w: px
            - ts_*: ms
            - fps: frames/s
            - hfov_deg: degrees
            - cone_height_m: meters
        Complexity: O(n) for number of tracks.
        Errors: logs warnings on queue failures.
        """
        if not self._enabled or not self._queue:
            return
        try:
            row = self._format_row(
                frame_idx=frame_idx,
                frame_w=frame_w,
                tracks=tracks or [],
                ts_wallclock_ms=ts_wallclock_ms,
                ts_source_ms=ts_source_ms,
                fps=fps,
                hfov_deg=hfov_deg,
                cone_height_m=cone_height_m,
            )
            self._queue.put_nowait(("row", row))
        except Exception as exc:
            log_event(
                logger,
                {},
                logging.WARNING,
                f"⚠️ Failed to queue CSV row: {exc}",
                "csv_queue_failed",
                error=str(exc),
            )

    def flush(self) -> None:
        """Flush queued CSV rows to disk.

        Complexity: O(1) for signaling; IO handled by worker.
        """
        if not self._enabled or not self._queue:
            return
        event = threading.Event()
        try:
            self._queue.put_nowait(("flush", event))
        except Exception:
            return
        event.wait(timeout=2.0)

    def close(self) -> None:
        """Close CSV logging and stop worker thread.

        Complexity: O(1) plus filesystem operations.
        Errors: logs warnings on failures.
        """
        with self._lock:
            if not self._queue:
                return
            self._enabled = False
            queue_ref = self._queue
            worker = self._worker
            handle = self._handle

        event = threading.Event()
        try:
            queue_ref.put_nowait(("close", event))
        except Exception:
            event.set()
        event.wait(timeout=2.0)
        if worker:
            worker.join(timeout=2.0)
        if handle and not handle.closed:
            try:
                handle.close()
            except Exception:
                log_event(
                    logger,
                    {},
                    logging.WARNING,
                    "⚠️ Failed to close CSV log handle",
                    "csv_close_failed",
                )
        with self._lock:
            self._queue = None
            self._worker = None
            self._handle = None

    def _prepare_directory(self, path: str) -> None:
        try:
            os.makedirs(path, mode=0o755, exist_ok=True)
            try:
                os.chmod(path, 0o755)
            except OSError:
                pass
        except OSError as exc:
            log_event(
                logger,
                {},
                logging.ERROR,
                f"⚠️ Failed to create CSV log dir '{path}': {exc}",
                "csv_mkdir_failed",
                path=path,
                error=str(exc),
            )

    def _unique_filename(self, output_dir: str, filename: str) -> str:
        base, ext = os.path.splitext(filename)
        if ext.lower() != ".csv":
            ext = ".csv"
        candidate = f"{base}{ext}"
        counter = 1
        while os.path.exists(os.path.join(output_dir, candidate)) and counter < MAX_FILENAME_ATTEMPTS:
            candidate = f"{base}_{counter}{ext}"
            counter += 1
        if counter >= MAX_FILENAME_ATTEMPTS and os.path.exists(os.path.join(output_dir, candidate)):
            for _ in range(UUID_FILENAME_ATTEMPTS):
                candidate = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
                if not os.path.exists(os.path.join(output_dir, candidate)):
                    break
        return candidate

    def _worker_loop(self, queue_ref: queue.Queue) -> None:
        frames_since_flush = 0
        while True:
            try:
                item = queue_ref.get()
            except Exception:
                break
            kind = None
            payload = None
            try:
                kind, payload = item
                if kind == "row":
                    if self._handle:
                        self._handle.write(payload + "\n")
                        frames_since_flush += 1
                        if frames_since_flush >= self._flush_every_frames:
                            self._handle.flush()
                            frames_since_flush = 0
                elif kind == "flush":
                    if self._handle:
                        self._handle.flush()
                    if isinstance(payload, threading.Event):
                        payload.set()
                elif kind == "close":
                    if self._handle:
                        self._handle.flush()
                        self._handle.close()
                    if isinstance(payload, threading.Event):
                        payload.set()
                    break
            except Exception as exc:
                logger.error(f"⚠️ CSV writer error: {exc}")
                if kind in ("flush", "close") and isinstance(payload, threading.Event):
                    payload.set()
                if self._handle and not self._handle.closed:
                    try:
                        self._handle.close()
                    except Exception:
                        pass
                with self._lock:
                    self._enabled = False
                break
            finally:
                queue_ref.task_done()

    def _format_row(
        self,
        frame_idx: int,
        frame_w: int,
        tracks: Iterable["Track"],
        ts_wallclock_ms: int,
        ts_source_ms: int,
        fps: float,
        hfov_deg: Optional[float],
        cone_height_m: Optional[float],
    ) -> str:
        track_list = list(tracks)
        target = None
        if track_list:
            target = max(track_list, key=self._safe_score)
            target_score = self._safe_score(target)
            if target_score == float("-inf"):
                target = None

        detected = target is not None
        center_x = frame_w / 2.0 if frame_w > 0 else 0.0

        target_id = ""
        cx = cy = err_px = err_norm = err_deg = bbox_h = est_dist_m = avg_score = ""
        if target is not None:
            target_id = str(target.track_id)
            cx = f"{target.cx:.1f}"
            cy = f"{target.cy:.1f}"
            err_px_val = target.cx - center_x
            err_px = f"{err_px_val:+.1f}"
            if frame_w > 0:
                err_norm_val = clamp(err_px_val / (frame_w / 2.0), -1.0, 1.0)
                err_norm = f"{err_norm_val:+.3f}"
            focal_px = _focal_px(frame_w, hfov_deg)
            if focal_px:
                angle_deg = math.degrees(math.atan(err_px_val / focal_px))
                err_deg = f"{angle_deg:+.2f}"
                if cone_height_m is not None and target.h > MIN_BBOX_HEIGHT_FOR_DISTANCE:
                    est_dist = (cone_height_m * focal_px) / target.h
                    est_dist_m = f"{est_dist:.2f}"
            bbox_h = str(int(target.h))
            avg_score = f"{target.avg_score():.2f}"

        row = [
            str(int(frame_idx)),
            str(int(ts_wallclock_ms)),
            str(int(ts_source_ms)),
            self._source_label,
            str(bool(detected)).lower(),
            target_id,
            cx,
            cy,
            err_px,
            err_norm,
            err_deg,
            bbox_h,
            est_dist_m,
            avg_score,
            f"{fps:.2f}",
        ]
        return ",".join(row)

    @staticmethod
    def _safe_score(track) -> float:
        try:
            score = float(track.avg_score())
        except Exception:
            return float("-inf")
        return score if math.isfinite(score) else float("-inf")
