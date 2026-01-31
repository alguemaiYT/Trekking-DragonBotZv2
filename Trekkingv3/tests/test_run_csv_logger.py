import os
import sys
import time
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch
import math
import types

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.modules.setdefault("cv2", MagicMock())
def _mock_numpy_mean(seq):
    return sum(seq) / len(seq) if len(seq) > 0 else 0.0

sys.modules.setdefault(
    "numpy",
    types.SimpleNamespace(mean=_mock_numpy_mean, hypot=math.hypot, ndarray=object),
)

from cone_tracker.run_csv_logger import CSV_HEADER, RunCSVLogger
from cone_tracker.tracker import Track


def _make_config(tmp_path, **overrides):
    config = {
        "debug": {
            "csv_export": {
                "enabled": True,
                "output_dir": str(tmp_path),
                "filename_pattern": "run_{source}_{start_ts}.csv",
                "flush_every_frames": 1,
            },
            "cone_height_m": 0.28,
        },
        "camera": {
            "hfov_deg": 70.0,
        },
    }
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    return config


def _make_track(cx=150.0, cy=60.0, w=40.0, h=80.0, score=0.8):
    track = Track(track_id=1, score_hist=deque(maxlen=10))
    track.cx = cx
    track.cy = cy
    track.w = w
    track.h = h
    track.score_hist.append(score)
    return track


def _read_csv(path: Path):
    lines = path.read_text().strip().splitlines()
    header = lines[0].split(",")
    rows = [line.split(",") for line in lines[1:]]
    return header, rows


def _wait_for_queue(logger: RunCSVLogger, timeout_s: float = 2.0):
    start = time.time()
    while time.time() - start < timeout_s:
        if logger._queue is None or logger._queue.unfinished_tasks == 0:
            return
        time.sleep(0.01)
    assert logger._queue is None or logger._queue.unfinished_tasks == 0


def test_csv_header_and_line(tmp_path):
    logger = RunCSVLogger()
    config = _make_config(tmp_path)
    csv_path = logger.open_if_enabled(config, "video", "2026-01-17T12:34:56.789Z")
    assert csv_path

    logger.log_frame(
        frame_idx=0,
        frame_w=200,
        tracks=[_make_track()],
        ts_wallclock_ms=1000,
        ts_source_ms=900,
        fps=30.0,
        hfov_deg=70.0,
        cone_height_m=0.28,
    )
    logger.flush()
    logger.close()

    header, rows = _read_csv(Path(csv_path))
    assert header == list(CSV_HEADER)
    assert rows
    assert rows[0][0] == "0"
    assert rows[0][3] == "video"
    assert rows[0][4] == "true"


def test_err_norm_formatting(tmp_path):
    logger = RunCSVLogger()
    config = _make_config(tmp_path)
    csv_path = logger.open_if_enabled(config, "camera", "2026-01-17T12:34:56.789Z")
    logger.log_frame(
        frame_idx=1,
        frame_w=200,
        tracks=[_make_track()],
        ts_wallclock_ms=1000,
        ts_source_ms=1000,
        fps=30.0,
        hfov_deg=70.0,
        cone_height_m=0.28,
    )
    logger.close()

    _, rows = _read_csv(Path(csv_path))
    assert rows[0][9] == "+0.500"


def test_invalid_hfov_omits_err_deg(tmp_path):
    logger = RunCSVLogger()
    config = _make_config(tmp_path, camera={"hfov_deg": 5.0})
    csv_path = logger.open_if_enabled(config, "video", "2026-01-17T12:34:56.789Z")
    logger.log_frame(
        frame_idx=0,
        frame_w=200,
        tracks=[_make_track()],
        ts_wallclock_ms=1000,
        ts_source_ms=900,
        fps=30.0,
        hfov_deg=5.0,
        cone_height_m=0.28,
    )
    logger.close()

    _, rows = _read_csv(Path(csv_path))
    assert rows[0][10] == ""


def test_flush_every_frames(tmp_path):
    logger = RunCSVLogger()
    config = _make_config(tmp_path)
    csv_path = logger.open_if_enabled(config, "camera", "2026-01-17T12:34:56.789Z")
    assert csv_path

    with patch.object(logger._handle, "flush", wraps=logger._handle.flush) as flush_mock:
        logger.log_frame(
            frame_idx=0,
            frame_w=200,
            tracks=[_make_track()],
            ts_wallclock_ms=1000,
            ts_source_ms=1000,
            fps=30.0,
            hfov_deg=70.0,
            cone_height_m=0.28,
        )
        _wait_for_queue(logger)
        assert flush_mock.call_count >= 1
    logger.close()


def test_close(tmp_path):
    logger = RunCSVLogger()
    config = _make_config(tmp_path)
    csv_path = logger.open_if_enabled(config, "video", "2026-01-17T12:34:56.789Z")
    logger.log_frame(
        frame_idx=0,
        frame_w=200,
        tracks=[_make_track()],
        ts_wallclock_ms=1000,
        ts_source_ms=900,
        fps=30.0,
        hfov_deg=70.0,
        cone_height_m=0.28,
    )
    logger.close()
    assert logger._worker is None
    assert Path(csv_path).exists()
