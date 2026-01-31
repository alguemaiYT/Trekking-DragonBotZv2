"""
Unit tests for video output functionality in cone_tracker
Tests headless execution and video saving features
"""
import copy
import cv2
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
import pytest

from cone_tracker.config import DEFAULT_CONFIG, load_config


class TestVideoOutputConfig:
    """Tests for video output configuration"""
    
    def test_default_config_has_output_video_path(self):
        """Test that DEFAULT_CONFIG includes output_video_path field"""
        assert "camera" in DEFAULT_CONFIG
        assert "output_video_path" in DEFAULT_CONFIG["camera"]
        assert DEFAULT_CONFIG["camera"]["output_video_path"] == ""
    
    def test_default_config_output_video_path_is_empty_string(self):
        """Test that output_video_path defaults to empty string"""
        assert DEFAULT_CONFIG["camera"]["output_video_path"] == ""
    
    def test_load_config_preserves_output_video_path(self):
        """Test that load_config preserves output_video_path from YAML"""
        # Create a temporary config file with output_video_path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("camera:\n")
            f.write("  output_video_path: 'output_processed.mp4'\n")
            temp_config_path = f.name
        
        try:
            config = load_config(temp_config_path)
            assert config["camera"]["output_video_path"] == "output_processed.mp4"
        finally:
            os.unlink(temp_config_path)
    
    def test_load_config_uses_default_when_no_output_video_path(self):
        """Test that load_config uses default empty string when output_video_path not in YAML"""
        # Create a temporary config file without output_video_path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("camera:\n")
            f.write("  index: 0\n")
            temp_config_path = f.name
        
        try:
            config = load_config(temp_config_path)
            assert config["camera"]["output_video_path"] == ""
        finally:
            os.unlink(temp_config_path)


class TestVideoOutputLogic:
    """Tests for video output logic in App.run()"""
    
    @patch('cone_tracker.app.cv2.VideoWriter')
    @patch('cone_tracker.app.cv2.VideoCapture')
    @patch('cone_tracker.app.os.path.exists')
    @patch('cone_tracker.app.load_config')
    def test_creates_video_writer_when_output_path_configured(
        self, mock_load_config, mock_exists, mock_video_capture, mock_video_writer
    ):
        """Test that VideoWriter is created when output_video_path is configured"""
        # Mock config with output path
        mock_config = copy.deepcopy(DEFAULT_CONFIG)
        mock_config["camera"]["output_video_path"] = "output.mp4"
        mock_config["debug"]["show_windows"] = False
        mock_load_config.return_value = mock_config
        
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # End immediately
        mock_video_capture.return_value = mock_cap
        
        # Mock VideoWriter
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_video_writer.return_value = mock_writer
        
        from cone_tracker import App
        app = App()
        
        try:
            app.run()
        except Exception:
            pass  # Expected to fail when reading frames
        
        # Verify VideoWriter was created with correct parameters
        mock_video_writer.assert_called_once()
        call_args = mock_video_writer.call_args
        assert call_args[0][0] == "output.mp4"  # output path
        assert call_args[0][2] == mock_config["camera"]["fps"]  # fps from config
        assert call_args[0][3] == (960, 540)  # size from process_width/height
    
    @patch('cone_tracker.app.cv2.VideoWriter')
    @patch('cone_tracker.app.cv2.VideoCapture')
    @patch('cone_tracker.app.load_config')
    def test_no_video_writer_when_output_path_empty(
        self, mock_load_config, mock_video_capture, mock_video_writer
    ):
        """Test that VideoWriter is not created when output_video_path is empty"""
        # Mock config with empty output path
        mock_config = copy.deepcopy(DEFAULT_CONFIG)
        mock_config["camera"]["output_video_path"] = ""
        mock_config["debug"]["show_windows"] = False
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
            pass  # Expected to fail when reading frames
        
        # Verify VideoWriter was NOT created
        mock_video_writer.assert_not_called()
    
    @patch('cone_tracker.app.cv2.VideoWriter')
    @patch('cone_tracker.app.cv2.VideoCapture')
    @patch('cone_tracker.app.os.path.exists')
    @patch('cone_tracker.app.load_config')
    def test_video_writer_release_called_on_exit(
        self, mock_load_config, mock_exists, mock_video_capture, mock_video_writer
    ):
        """Test that VideoWriter.release() is called when exiting"""
        # Mock config with output path
        mock_config = copy.deepcopy(DEFAULT_CONFIG)
        mock_config["camera"]["output_video_path"] = "output.mp4"
        mock_config["debug"]["show_windows"] = False
        mock_load_config.return_value = mock_config
        
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # End immediately
        mock_video_capture.return_value = mock_cap
        
        # Mock VideoWriter
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_video_writer.return_value = mock_writer
        
        from cone_tracker import App
        app = App()
        
        try:
            app.run()
        except Exception:
            pass
        
        # Verify both cap.release() and writer.release() were called
        mock_cap.release.assert_called_once()
        mock_writer.release.assert_called_once()


class TestHeadlessExecution:
    """Tests for headless execution (no GUI)"""
    
    @patch('cone_tracker.app.cv2.imshow')
    @patch('cone_tracker.app.cv2.VideoCapture')
    @patch('cone_tracker.app.load_config')
    def test_handles_cv2_error_gracefully(
        self, mock_load_config, mock_video_capture, mock_imshow
    ):
        """Test that cv2.error from imshow is caught and handled gracefully"""
        import numpy as np
        
        # Mock config with show_windows enabled
        mock_config = copy.deepcopy(DEFAULT_CONFIG)
        mock_config["debug"]["show_windows"] = True
        mock_load_config.return_value = mock_config
        
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Return one valid frame, then fail
        fake_frame = np.zeros((540, 960, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, fake_frame), (False, None)]
        mock_video_capture.return_value = mock_cap
        
        # Mock cv2.imshow to raise cv2.error (headless environment)
        mock_imshow.side_effect = cv2.error("The function is not implemented")
        
        from cone_tracker import App
        app = App()
        
        # Should not raise exception, should handle gracefully
        try:
            app.run()
        except cv2.error:
            pytest.fail("cv2.error should be caught and handled gracefully")
        except Exception:
            pass  # Other exceptions are ok (e.g., from processing)
    
    @patch('cone_tracker.app.cv2.destroyAllWindows')
    @patch('cone_tracker.app.cv2.VideoCapture')
    @patch('cone_tracker.app.load_config')
    def test_handles_destroyAllWindows_error_gracefully(
        self, mock_load_config, mock_video_capture, mock_destroy
    ):
        """Test that errors from destroyAllWindows are caught in finally block"""
        # Mock config
        mock_config = copy.deepcopy(DEFAULT_CONFIG)
        mock_config["debug"]["show_windows"] = False
        mock_load_config.return_value = mock_config
        
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap
        
        # Mock cv2.destroyAllWindows to raise error
        mock_destroy.side_effect = Exception("No GUI available")
        
        from cone_tracker import App
        app = App()
        
        # Should not raise exception from destroyAllWindows
        try:
            app.run()
        except Exception as e:
            if "No GUI available" in str(e):
                pytest.fail("destroyAllWindows exception should be caught in finally block")
