import copy
from unittest.mock import MagicMock, call, patch

import numpy as np

from cone_tracker.config import DEFAULT_CONFIG

MIN_RGB_SHAPE = (1, 1, 3)


def _gui_config():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["camera"]["capture_width"] = 320
    config["camera"]["capture_height"] = 240
    config["camera"]["process_width"] = 160
    config["camera"]["process_height"] = 120
    config["camera"]["max_consecutive_read_failures"] = 1
    config["debug"]["show_windows"] = True
    config["debug"]["show_mask"] = True
    return config


def _fake_capture(frame):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, frame), (False, None)]
    return mock_cap


def _fake_detector(mask):
    detector = MagicMock()
    detector.detect.return_value = ([], mask, [])
    return detector


def test_gui_init_and_waitkey_called():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    mask = np.zeros((540, 960), dtype=np.uint8)
    config = _gui_config()

    with patch("cone_tracker.app.load_config", return_value=config), \
        patch("cone_tracker.app.os.environ", {"DISPLAY": ":0"}), \
        patch("cone_tracker.app.cv2.VideoCapture", return_value=_fake_capture(frame)), \
        patch("cone_tracker.app.ConeDetector", return_value=_fake_detector(mask)), \
        patch("cone_tracker.app.cv2.namedWindow") as named_window, \
        patch("cone_tracker.app.cv2.imshow") as imshow, \
        patch("cone_tracker.app.cv2.waitKey", return_value=ord("q")) as wait_key:
        from cone_tracker import App

        app = App()
        app.run()

    assert named_window.call_count >= 2
    tracker_calls = [args for args, _ in imshow.call_args_list if args and args[0] == "Tracker"]
    assert any(call_args[1].shape == MIN_RGB_SHAPE for call_args in tracker_calls)
    wait_key.assert_called()


def test_get_mask_returns_uint8():
    from cone_tracker.detector import ConeDetector

    config = _gui_config()
    detector = ConeDetector(config)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv = detector.preprocess(frame)
    mask = detector.get_mask(frame, hsv)
    assert mask.dtype == np.uint8
    assert mask.max() in (0, 255)


def test_hot_reload_recreates_windows():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    mask = np.zeros((540, 960), dtype=np.uint8)
    config = _gui_config()
    updated = copy.deepcopy(config)
    updated["debug"]["show_mask"] = False

    with patch("cone_tracker.app.load_config", side_effect=[config, updated]), \
        patch("cone_tracker.app.os.environ", {"DISPLAY": ":0"}), \
        patch("cone_tracker.app.cv2.VideoCapture", return_value=_fake_capture(frame)), \
        patch("cone_tracker.app.ConeDetector", return_value=_fake_detector(mask)), \
        patch("cone_tracker.app.cv2.namedWindow") as named_window, \
        patch("cone_tracker.app.cv2.imshow") as imshow, \
        patch("cone_tracker.app.cv2.waitKey", return_value=ord("q")), \
        patch("cone_tracker.app.cv2.destroyWindow") as destroy_window:
        from cone_tracker import App

        app = App()
        app.reload_config()

    assert call("Mask") in destroy_window.call_args_list
    assert call("Tracker") in named_window.call_args_list


def test_visualizer_mask_normalization():
    from cone_tracker.visualizer import Visualizer

    config = _gui_config()
    vis = Visualizer(config)
    mask_bool = np.array([[True, False]], dtype=bool)
    mask_float = np.array([[0.5, 1.0]], dtype=np.float32)
    assert vis._normalize_mask(mask_bool).dtype == np.uint8
    assert vis._normalize_mask(mask_float).dtype == np.uint8
