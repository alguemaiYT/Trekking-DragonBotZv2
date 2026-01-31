"""
Tests for structured run-log export behavior.
"""
import copy
import json
from unittest.mock import MagicMock, patch

import numpy as np

from cone_tracker import App
from cone_tracker.config import DEFAULT_CONFIG


def _base_config(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["camera"]["capture_width"] = 320
    config["camera"]["capture_height"] = 240
    config["camera"]["process_width"] = 160
    config["camera"]["process_height"] = 120
    config["camera"]["max_consecutive_read_failures"] = 1
    config["debug"]["show_windows"] = False
    config["debug"]["show_mask"] = False
    config["debug"]["export_run_log"] = True
    config["debug"]["run_log_dir"] = str(tmp_path)
    config["debug"]["run_log_filename_pattern"] = "run_{source}_{start_ts}.jsonl"
    return config


def _log_records(log_path):
    content = log_path.read_text().splitlines()
    return [json.loads(line) for line in content if line.strip()]


def test_export_run_log_camera(tmp_path):
    config = _base_config(tmp_path)
    config["camera"]["video_path"] = ""

    frame = np.zeros((config["camera"]["capture_height"], config["camera"]["capture_width"], 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame), (False, None)]

    detector = MagicMock()
    detector.detect.return_value = ([], np.zeros((config["camera"]["process_height"], config["camera"]["process_width"]), dtype=np.uint8), [])

    with patch("cone_tracker.app.load_config", return_value=config), \
        patch("cone_tracker.app.os.path.exists", return_value=False), \
        patch("cone_tracker.app.cv2.VideoCapture", return_value=mock_cap), \
        patch("cone_tracker.app.watch_config", return_value=False), \
        patch("cone_tracker.app.ConeDetector", return_value=detector):
        app = App()
        app.run()

    log_files = list(tmp_path.glob("*.jsonl"))
    assert len(log_files) == 1

    records = _log_records(log_files[0])
    assert records
    record = records[0]
    assert record["frame_idx"] == 0
    assert record["source"] == "camera"
    assert record["ts_source_ms"] == record["ts_wallclock_ms"]


def test_export_run_log_video_timestamp(tmp_path):
    config = _base_config(tmp_path)
    config["camera"]["video_path"] = "sample.mp4"
    config["camera"]["index"] = -1

    frame = np.zeros((config["camera"]["capture_height"], config["camera"]["capture_width"], 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    end_of_stream = (False, None)
    mock_cap.read.side_effect = [(True, frame), (True, frame), end_of_stream, end_of_stream]
    mock_cap.get.return_value = 1000.0

    detector = MagicMock()
    detector.detect.return_value = ([], np.zeros((config["camera"]["process_height"], config["camera"]["process_width"]), dtype=np.uint8), [])

    with patch("cone_tracker.app.load_config", return_value=config), \
        patch("cone_tracker.app.os.path.exists", return_value=True), \
        patch("cone_tracker.app.cv2.VideoCapture", return_value=mock_cap), \
        patch("cone_tracker.app.watch_config", return_value=False), \
        patch("cone_tracker.app.ConeDetector", return_value=detector):
        app = App()
        app.run()

    log_files = list(tmp_path.glob("*.jsonl"))
    assert len(log_files) == 1

    records = _log_records(log_files[0])
    assert records
    assert records[0]["source"] == "video"
    assert records[0]["ts_source_ms"] == 1000
    if len(records) > 1:
        assert records[1]["frame_idx"] == 1
