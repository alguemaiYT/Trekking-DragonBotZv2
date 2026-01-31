"""
Unit tests for video input functionality in cone_tracker
"""
import copy
import cv2
import logging
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import pytest

from cone_tracker.config import DEFAULT_CONFIG, load_config


class TestVideoInputConfig:
    """Tests for video input configuration"""
    
    def test_default_config_has_video_path(self):
        """Test that DEFAULT_CONFIG includes video_path field"""
        assert "camera" in DEFAULT_CONFIG
        assert "video_path" in DEFAULT_CONFIG["camera"]
        assert DEFAULT_CONFIG["camera"]["video_path"] == ""
    
    def test_default_config_video_path_is_empty_string(self):
        """Test that video_path defaults to empty string"""
        assert DEFAULT_CONFIG["camera"]["video_path"] == ""
    
    def test_load_config_preserves_video_path(self):
        """Test that load_config preserves video_path from YAML"""
        # Create a temporary config file with video_path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("camera:\n")
            f.write("  video_path: 'test_video.mp4'\n")
            temp_config_path = f.name
        
        try:
            config = load_config(temp_config_path)
            assert config["camera"]["video_path"] == "test_video.mp4"
        finally:
            os.unlink(temp_config_path)
    
    def test_load_config_uses_default_when_no_video_path(self):
        """Test that load_config uses default empty string when video_path not in YAML"""
        # Create a temporary config file without video_path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("camera:\n")
            f.write("  index: 0\n")
            temp_config_path = f.name
        
        try:
            config = load_config(temp_config_path)
            assert config["camera"]["video_path"] == ""
        finally:
            os.unlink(temp_config_path)


class TestVideoInputLogic:
    """Tests for video input logic in App.run()"""
    
    @patch('cone_tracker.app.cv2.VideoCapture')
    @patch('cone_tracker.app.os.path.exists')
    @patch('cone_tracker.app.load_config')
    def test_uses_video_when_path_exists(self, mock_load_config, mock_exists, mock_video_capture):
        """Test that app uses video file when path exists"""
        # Mock config with video path
        mock_config = copy.deepcopy(DEFAULT_CONFIG)
        mock_config["camera"]["video_path"] = "test_video.mp4"
        mock_config["camera"]["index"] = -1
        mock_config["camera"]["max_consecutive_read_failures"] = 1
        mock_load_config.return_value = mock_config
        
        # Mock video file exists
        mock_exists.return_value = True
        
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # End immediately
        mock_video_capture.return_value = mock_cap
        
        from cone_tracker import App
        app = App()
        
        try:
            app.run()
        except Exception:
            pass  # Expected to fail when trying to read frames
        
        # Verify VideoCapture was called with video path (not camera index)
        mock_video_capture.assert_called_with("test_video.mp4")

    @patch('cone_tracker.app.os.path.exists')
    @patch('cone_tracker.app.load_config')
    def test_conflicting_camera_and_video_raises(self, mock_load_config, mock_exists):
        """Test that app refuses to start when both camera and video are set"""
        mock_config = copy.deepcopy(DEFAULT_CONFIG)
        mock_config["camera"]["video_path"] = "test_video.mp4"
        mock_config["camera"]["index"] = 0
        mock_load_config.return_value = mock_config
        mock_exists.return_value = True

        from cone_tracker import App
        with pytest.raises(SystemExit) as excinfo:
            App()
        assert "CONFIG ERROR: Both camera.index and camera.video_path are set." in excinfo.value.args[0]

    @patch('cone_tracker.app.cv2.VideoCapture')
    @patch('cone_tracker.app.os.path.exists')
    @patch('cone_tracker.app.load_config')
    def test_uses_camera_when_video_path_empty(self, mock_load_config, mock_exists, mock_video_capture):
        """Test that app uses camera when video_path is empty"""
        # Mock config with empty video path
        mock_config = copy.deepcopy(DEFAULT_CONFIG)
        mock_config["camera"]["video_path"] = ""
        mock_config["camera"]["max_consecutive_read_failures"] = 1
        mock_load_config.return_value = mock_config
        
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # End immediately
        mock_video_capture.return_value = mock_cap
        
        from cone_tracker import App
        app = App()
        
        try:
            app.run()
        except Exception:
            pass  # Expected to fail when trying to read frames
        
        # Verify VideoCapture was called with camera index and V4L2 backend
        mock_video_capture.assert_called_with(0, cv2.CAP_V4L2)
    
    @patch('cone_tracker.app.cv2.VideoCapture')
    @patch('cone_tracker.app.os.path.exists')
    @patch('cone_tracker.app.load_config')
    def test_fallback_to_camera_when_video_not_found(self, mock_load_config, mock_exists, mock_video_capture, caplog):
        """Test that app falls back to camera when video file doesn't exist"""
        # Mock config with video path
        mock_config = copy.deepcopy(DEFAULT_CONFIG)
        mock_config["camera"]["video_path"] = "nonexistent_video.mp4"
        mock_config["camera"]["max_consecutive_read_failures"] = 1
        mock_load_config.return_value = mock_config
        
        # Mock video file does not exist
        mock_exists.return_value = False
        
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # End immediately
        mock_video_capture.return_value = mock_cap
        
        from cone_tracker import App
        app = App()
        
        try:
            with caplog.at_level(logging.WARNING):
                app.run()
        except Exception:
            pass  # Expected to fail when trying to read frames

        assert "VIDEO PATH configured but file not found -> falling back to camera" in caplog.text
        
        # Verify VideoCapture was called with camera (fallback), not video path
        mock_video_capture.assert_called_with(0, cv2.CAP_V4L2)
