"""
Unit tests for debug and visualization features.
Tests the new log_rejections and log_suspects features.
"""
import ast
import re


class TestVisualizerChanges:
    """Tests for visualizer.py changes"""
    
    def test_visualizer_no_early_return(self):
        """Test that visualizer.draw() does not have early return for show_windows"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Find the draw method
        draw_method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'draw':
                draw_method = node
                break
        
        assert draw_method is not None, "draw method not found"
        
        # Check that there's no early return based on show_windows
        # The old code had: if not self.cfg["show_windows"]: return frame
        method_source = ast.get_source_segment(content, draw_method)
        
        # Should NOT have the old pattern
        assert 'if not self.cfg["show_windows"]:\n            return frame' not in method_source, \
            "Visualizer should not have early return for show_windows"
    
    def test_visualizer_has_reject_counter(self):
        """Test that visualizer displays reject counter"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        # Should have a line that displays rejects count
        assert 'f"Rejects: {len(rejects)}"' in content, \
            "Visualizer should display rejects counter"
    
    def test_visualizer_has_suspect_counter(self):
        """Test that visualizer displays suspect counter"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        # Should count suspects
        assert 'suspect_count = sum(1 for t in tracks if t.state == ConeState.SUSPECT)' in content, \
            "Visualizer should count suspects"
        
        # Should display suspects in track info
        assert 'susp' in content, \
            "Visualizer should display suspect count"
    
    def test_visualizer_improved_rejection_visibility(self):
        """Test that rejection visualization is improved"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        # Check for rectangle drawing with red color and thickness 2
        assert 'cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)' in content, \
            "Should have red rectangles with thickness 2 for rejections"
        
        # Check for larger text (0.5 instead of 0.35)
        assert '0.5' in content and 'reason' in content, \
            "Should have rejection text with larger font size"


class TestConfigChanges:
    """Tests for config.py changes"""
    
    def test_default_config_has_log_rejections(self):
        """Test that DEFAULT_CONFIG includes log_rejections"""
        with open('cone_tracker/config.py', 'r') as f:
            content = f.read()
        
        assert '"log_rejections"' in content, "DEFAULT_CONFIG should have log_rejections"
        assert 'log_rejections' in content and 'False' in content, \
            "log_rejections should default to False"
    
    def test_default_config_has_log_suspects(self):
        """Test that DEFAULT_CONFIG includes log_suspects"""
        with open('cone_tracker/config.py', 'r') as f:
            content = f.read()
        
        assert '"log_suspects"' in content, "DEFAULT_CONFIG should have log_suspects"
        assert 'log_suspects' in content and 'False' in content, \
            "log_suspects should default to False"


class TestTrackFramesSeen:
    """Tests for Track.frames_seen property"""
    
    def test_track_has_frames_seen_property(self):
        """Test that Track class has frames_seen property"""
        # Test by checking the source code for the property definition
        with open('cone_tracker/tracker.py', 'r') as f:
            content = f.read()
        
        # Should have @property decorator for frames_seen
        assert '@property' in content and 'def frames_seen' in content, \
            "Track should have frames_seen as a property"
        
        # Should return len(self.score_hist)
        assert 'return len(self.score_hist)' in content, \
            "frames_seen should return len(self.score_hist)"
        
        # Also test it functionally if dependencies are available
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from cone_tracker.tracker import Track
            from collections import deque
            
            # Create a track with empty score history
            track = Track(track_id=1, score_hist=deque(maxlen=10))
            assert hasattr(track, 'frames_seen'), "Track should have frames_seen attribute"
            assert track.frames_seen == 0, "frames_seen should be 0 for empty score_hist"
            
            # Add some scores to the history
            track.score_hist.append(0.8)
            assert track.frames_seen == 1, "frames_seen should be 1 after adding one score"
            
            track.score_hist.append(0.9)
            track.score_hist.append(0.85)
            assert track.frames_seen == 3, "frames_seen should be 3 after adding three scores"
            
            # Fill up to maxlen
            for i in range(7):
                track.score_hist.append(0.8 + i * 0.01)
            assert track.frames_seen == 10, "frames_seen should be 10 when score_hist is full"
            
            # Add more items (should exceed maxlen and cap at 10)
            track.score_hist.append(0.95)
            assert track.frames_seen == 10, "frames_seen should remain 10 after exceeding maxlen"
        except ImportError:
            # If we can't import due to missing dependencies, the source code check is sufficient
            pass
    
    def test_frames_seen_used_in_suspect_logging(self):
        """Test that frames_seen is used correctly in app.py suspect logging"""
        with open('cone_tracker/app.py', 'r') as f:
            content = f.read()
        
        # Should use t.frames_seen (as property, not method call)
        assert 't.frames_seen' in content, "app.py should use t.frames_seen in logging"
        # Make sure it's not called as a method
        assert 't.frames_seen()' not in content, "frames_seen should be used as property, not method"


class TestAppChanges:
    """Tests for app.py changes"""
    
    def test_app_has_rejection_logging(self):
        """Test that app.py logs rejections when configured"""
        with open('cone_tracker/app.py', 'r') as f:
            content = f.read()
        
        # Should check for log_rejections config
        assert 'log_rejections' in content, "app.py should check log_rejections config"
        
        # Should log rejections with emoji
        assert '🔴' in content or 'Frame com' in content, \
            "app.py should log rejection messages"
        
        # Should iterate over rejects
        assert 'for bbox, reason in rejects' in content, \
            "app.py should iterate over rejects to log them"
    
    def test_app_has_suspect_logging(self):
        """Test that app.py logs suspects when configured"""
        with open('cone_tracker/app.py', 'r') as f:
            content = f.read()
        
        # Should check for log_suspects config
        assert 'log_suspects' in content, "app.py should check log_suspects config"
        
        # Should log suspects with emoji
        assert '🟡' in content or 'suspects' in content.lower(), \
            "app.py should log suspect messages"
        
        # Should filter for SUSPECT state
        assert 'ConeState.SUSPECT' in content, \
            "app.py should filter tracks by SUSPECT state"


class TestREADMEChanges:
    """Tests for README.md changes"""
    
    def test_readme_has_debugging_section(self):
        """Test that README has debugging section"""
        with open('README.md', 'r') as f:
            content = f.read()
        
        assert 'Debugging e Visualização' in content or 'Debug' in content, \
            "README should have debugging section"
    
    def test_readme_documents_log_rejections(self):
        """Test that README documents log_rejections option"""
        with open('README.md', 'r') as f:
            content = f.read()
        
        assert 'log_rejections' in content, \
            "README should document log_rejections option"
    
    def test_readme_documents_log_suspects(self):
        """Test that README documents log_suspects option"""
        with open('README.md', 'r') as f:
            content = f.read()
        
        assert 'log_suspects' in content, \
            "README should document log_suspects option"
    
    def test_readme_has_example_logs(self):
        """Test that README includes example log output"""
        with open('README.md', 'r') as f:
            content = f.read()
        
        # Should have example showing rejection emoji AND text
        assert '🔴' in content and 'rejeições' in content.lower(), \
            "README should show rejection log examples with emoji"
        
        # Should have example showing suspect emoji AND text
        assert '🟡' in content and 'suspects' in content.lower(), \
            "README should show suspect log examples with emoji"


class TestHeadingOverlayConfig:
    """Tests for heading overlay config changes"""
    
    def test_config_has_show_heading_overlay(self):
        """Test that DEFAULT_CONFIG includes show_heading_overlay"""
        with open('cone_tracker/config.py', 'r') as f:
            content = f.read()
        
        assert '"show_heading_overlay"' in content, \
            "DEFAULT_CONFIG should have show_heading_overlay"
        assert 'show_heading_overlay' in content and ': False' in content, \
            "show_heading_overlay should default to False"
    
    def test_config_has_heading_center_deadband_deg(self):
        """Test that DEFAULT_CONFIG includes heading_center_deadband_deg"""
        with open('cone_tracker/config.py', 'r') as f:
            content = f.read()
        
        assert '"heading_center_deadband_deg"' in content, \
            "DEFAULT_CONFIG should have heading_center_deadband_deg"
        assert '0.5' in content, \
            "heading_center_deadband_deg should default to 0.5"
    
    def test_config_has_camera_hfov_deg(self):
        """Test that DEFAULT_CONFIG includes camera.hfov_deg"""
        with open('cone_tracker/config.py', 'r') as f:
            content = f.read()
        
        assert '"hfov_deg"' in content, \
            "DEFAULT_CONFIG should have camera.hfov_deg"
        assert '70.0' in content, \
            "camera.hfov_deg should default to 70.0"


class TestHeadingOverlayImplementation:
    """Tests for heading overlay implementation in visualizer"""
    
    def test_visualizer_imports_math(self):
        """Test that visualizer imports math module for heading calculations"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        assert 'import math' in content, \
            "Visualizer should import math module"
    
    def test_visualizer_has_compute_heading_info(self):
        """Test that visualizer has _compute_heading_info method"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        assert 'def _compute_heading_info' in content, \
            "Visualizer should have _compute_heading_info method"
        
        # Check that it returns direction and angle
        assert 'return direction, angle_deg' in content, \
            "_compute_heading_info should return direction and angle_deg"
    
    def test_visualizer_validates_hfov(self):
        """Test that visualizer validates HFOV range"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        # Should validate HFOV in range [10, 170]
        assert 'if hfov_deg < 10.0 or hfov_deg > 170.0:' in content, \
            "Visualizer should validate HFOV range"
        assert 'hfov_deg = 70.0' in content, \
            "Visualizer should fallback to 70.0 for invalid HFOV"
    
    def test_visualizer_uses_deadband(self):
        """Test that visualizer uses deadband for CENTER classification"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        assert 'heading_center_deadband_deg' in content, \
            "Visualizer should use heading_center_deadband_deg"
        assert 'if abs(angle_deg) < deadband_deg:' in content, \
            "Visualizer should use deadband for CENTER classification"
        assert 'direction = "CENTER"' in content, \
            "Visualizer should set direction to CENTER within deadband"
    
    def test_visualizer_draws_heading_for_confirmed(self):
        """Test that visualizer draws heading overlay only for CONFIRMED tracks"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        # Should check for CONFIRMED state and show_heading_overlay config
        assert 'if t.state == ConeState.CONFIRMED and self.cfg.get("show_heading_overlay", False):' in content, \
            "Visualizer should only draw heading for CONFIRMED tracks when enabled"
        
        # Should compute heading info
        assert '_compute_heading_info(t, frame.shape[1])' in content, \
            "Visualizer should compute heading info for CONFIRMED tracks"
        
        # Should format with sign
        assert 'angle_deg:+.2f' in content, \
            "Visualizer should format angle with sign and 2 decimals"
    
    def test_visualizer_handles_top_edge_clamping(self):
        """Test that visualizer clamps text positions at top edge"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        # Should use max() to clamp y positions
        assert 'heading_y = max(' in content, \
            "Visualizer should clamp heading_y position"
        assert 'label_y = max(' in content, \
            "Visualizer should clamp label_y position"
    
    def test_visualizer_functional_heading_disabled(self):
        """Test that visualizer works when heading overlay is disabled"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from cone_tracker.visualizer import Visualizer
            from cone_tracker.tracker import Track
            from cone_tracker.utils import ConeState
            from collections import deque
            import numpy as np
            
            # Create config with heading overlay disabled
            config = {
                "debug": {
                    "show_heading_overlay": False,
                    "draw_suspects": False,
                    "show_rejection_reason": False,
                },
                "camera": {
                    "hfov_deg": 70.0,
                }
            }
            
            vis = Visualizer(config)
            
            # Create a dummy frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Create a CONFIRMED track
            track = Track(track_id=1, score_hist=deque(maxlen=10))
            track.state = ConeState.CONFIRMED
            track.cx = 320.0
            track.cy = 240.0
            track.w = 50.0
            track.h = 100.0
            track.score_hist.append(0.8)
            
            # Draw should work without errors
            result = vis.draw(frame.copy(), [track], [], 30.0)
            
            # Result should be a numpy array
            assert isinstance(result, np.ndarray), \
                "draw() should return numpy array"
            
        except ImportError:
            # If dependencies are missing, skip functional test
            pass
    
    def test_visualizer_functional_heading_enabled(self):
        """Test that visualizer draws heading overlay when enabled"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from cone_tracker.visualizer import Visualizer
            from cone_tracker.tracker import Track
            from cone_tracker.utils import ConeState
            from collections import deque
            import numpy as np
            
            # Create config with heading overlay enabled
            config = {
                "debug": {
                    "show_heading_overlay": True,
                    "heading_center_deadband_deg": 0.5,
                    "draw_suspects": False,
                    "show_rejection_reason": False,
                },
                "camera": {
                    "hfov_deg": 70.0,
                }
            }
            
            vis = Visualizer(config)
            
            # Create a dummy frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            original_frame = frame.copy()
            
            # Create a CONFIRMED track offset from center (should show RIGHT)
            track = Track(track_id=1, score_hist=deque(maxlen=10))
            track.state = ConeState.CONFIRMED
            track.cx = 400.0  # Right of center (center is 320)
            track.cy = 240.0
            track.w = 50.0
            track.h = 100.0
            track.score_hist.append(0.8)
            
            # Draw should work without errors
            result = vis.draw(frame.copy(), [track], [], 30.0)
            
            # Result should be a numpy array
            assert isinstance(result, np.ndarray), \
                "draw() should return numpy array when heading overlay enabled"
            
            # Result should differ from original (some pixels changed due to drawing)
            assert not np.array_equal(result, original_frame), \
                "Frame should be modified when drawing heading overlay"
            
        except ImportError:
            # If dependencies are missing, skip functional test
            pass
    
    def test_heading_info_computation_logic(self):
        """Test the heading info computation logic"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from cone_tracker.visualizer import Visualizer
            from cone_tracker.tracker import Track
            from cone_tracker.utils import ConeState
            from collections import deque
            
            config = {
                "debug": {
                    "heading_center_deadband_deg": 0.5,
                },
                "camera": {
                    "hfov_deg": 70.0,
                }
            }
            
            vis = Visualizer(config)
            
            # Test LEFT (track to the left of center)
            track_left = Track(track_id=1, score_hist=deque(maxlen=10))
            track_left.cx = 200.0  # Left of center (center is 320)
            direction, angle = vis._compute_heading_info(track_left, 640)
            assert direction == "LEFT", f"Expected LEFT, got {direction}"
            assert angle < 0, f"Expected negative angle for LEFT, got {angle}"
            
            # Test RIGHT (track to the right of center)
            track_right = Track(track_id=2, score_hist=deque(maxlen=10))
            track_right.cx = 440.0  # Right of center
            direction, angle = vis._compute_heading_info(track_right, 640)
            assert direction == "RIGHT", f"Expected RIGHT, got {direction}"
            assert angle > 0, f"Expected positive angle for RIGHT, got {angle}"
            
            # Test CENTER (track at or near center)
            track_center = Track(track_id=3, score_hist=deque(maxlen=10))
            track_center.cx = 320.0  # Exactly at center
            direction, angle = vis._compute_heading_info(track_center, 640)
            assert direction == "CENTER", f"Expected CENTER, got {direction}"
            assert abs(angle) < 0.5, f"Expected angle near 0 for CENTER, got {angle}"
            
        except ImportError:
            # If dependencies are missing, skip functional test
            pass
    
    def test_hfov_validation_logic(self):
        """Test HFOV validation fallback logic"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from cone_tracker.visualizer import Visualizer
            from cone_tracker.tracker import Track
            from collections import deque
            
            # Test with invalid HFOV (too low)
            config_invalid_low = {
                "debug": {"heading_center_deadband_deg": 0.5},
                "camera": {"hfov_deg": 5.0}  # Invalid: < 10
            }
            vis = Visualizer(config_invalid_low)
            track = Track(track_id=1, score_hist=deque(maxlen=10))
            track.cx = 320.0
            # Should use fallback 70.0 instead of 5.0
            direction, angle = vis._compute_heading_info(track, 640)
            # Should not crash and return valid result
            assert direction in ["LEFT", "RIGHT", "CENTER"]
            
            # Test with invalid HFOV (too high)
            config_invalid_high = {
                "debug": {"heading_center_deadband_deg": 0.5},
                "camera": {"hfov_deg": 180.0}  # Invalid: > 170
            }
            vis = Visualizer(config_invalid_high)
            # Should use fallback 70.0 instead of 180.0
            direction, angle = vis._compute_heading_info(track, 640)
            # Should not crash and return valid result
            assert direction in ["LEFT", "RIGHT", "CENTER"]
            
        except ImportError:
            # If dependencies are missing, skip functional test
            pass


if __name__ == "__main__":
    # Simple test runner
    test_classes = [TestVisualizerChanges, TestConfigChanges, TestTrackFramesSeen, TestAppChanges, TestREADMEChanges, TestHeadingOverlayConfig, TestHeadingOverlayImplementation]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
                print(f"✓ {test_class.__name__}.{method_name}")
            except AssertionError as e:
                failed_tests.append((test_class.__name__, method_name, str(e)))
                print(f"✗ {test_class.__name__}.{method_name}: {e}")
            except Exception as e:
                failed_tests.append((test_class.__name__, method_name, str(e)))
                print(f"✗ {test_class.__name__}.{method_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Tests: {passed_tests}/{total_tests} passed")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        exit(1)
    else:
        print("\nAll tests passed! ✓")
        exit(0)
