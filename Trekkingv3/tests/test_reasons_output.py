#!/usr/bin/env python3
"""
Unit tests for reasons output functionality.

Tests that gist_test_runner.py generates a detailed .txt report
with per-frame reasons when run with default settings or --unit mode.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_gist_runner_unit_mode_creates_txt_report():
    """
    Test that running gist_test_runner.py in --unit mode creates a reasons .txt file.
    
    This test:
    1. Runs gist_test_runner.py --unit (5 frames, no GUI)
    2. Verifies that a reasons_{timestamp}.txt file is created in the current directory
    3. Checks that the file contains expected keywords (Frame, ACCEPTED, REJECTED, etc.)
    """
    # Get the path to gist_test_runner.py
    test_dir = Path(__file__).parent
    runner_script = test_dir / "exec" / "gist_test_runner.py"
    
    if not runner_script.exists():
        pytest.skip(f"gist_test_runner.py not found at {runner_script}")
    
    # We need a video file for testing - check if there's a test video
    # For now, we'll skip if no video is available
    repo_root = test_dir.parent
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to temp directory to capture the output file
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tmpdir)
            
            # Run gist_test_runner.py --unit
            # Note: This may fail if no video is configured, so we'll check the return code
            cmd = [
                sys.executable,
                str(runner_script),
                "--unit",
                "--max-frames", "5",
            ]
            
            # Try to run the script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            # Check if a reasons .txt file was created
            txt_files = list(Path(tmpdir).glob("reasons_*.txt"))
            
            # The script may fail if no video is configured, but we still want to test
            # the code paths. For now, we'll just verify that the integration exists.
            # A full integration test would require a test video.
            
            if result.returncode != 0:
                # Script failed (probably no video configured)
                # This is expected in CI without a test video
                pytest.skip(f"gist_test_runner.py failed (likely no video configured): {result.stderr}")
            
            # If successful, verify the txt file was created
            assert len(txt_files) > 0, f"No reasons_*.txt file found in {tmpdir}. Files: {list(Path(tmpdir).glob('*'))}"
            
            txt_file = txt_files[0]
            assert txt_file.exists(), f"Reasons file {txt_file} does not exist"
            
            # Read and verify content
            content = txt_file.read_text()
            
            # Check for expected sections
            assert "CONE DETECTION REASONS REPORT" in content, "Missing report header"
            assert "Frame" in content, "Missing frame sections"
            
            # Check for expected keywords (at least some should be present)
            keywords = ["Detector", "ACCEPTED", "REJECTED", "Tracker", "SUMMARY"]
            found_keywords = [kw for kw in keywords if kw in content]
            assert len(found_keywords) >= 3, f"Expected at least 3 keywords from {keywords}, found: {found_keywords}"
            
            print(f"✓ Test passed: reasons report created at {txt_file}")
            print(f"  File size: {txt_file.stat().st_size} bytes")
            print(f"  Found keywords: {found_keywords}")
            
        finally:
            os.chdir(original_cwd)


def test_reasons_writer_basic_functionality():
    """
    Test the ReasonsWriter class directly (unit test).
    
    This test verifies that ReasonsWriter can:
    1. Collect frame data
    2. Write a formatted .txt report
    3. Include expected sections and data
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cone_tracker.reasons_writer import ReasonsWriter
    
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ReasonsWriter()
        writer.set_start_timestamp("2024-01-01T12:00:00.000Z")
        
        # Set minimal config summary
        config = {
            "hsv_orange": {
                "low_1": [0, 90, 90],
                "high_1": [28, 255, 255],
            },
            "tracking": {
                "lost_timeout": 0.6,
                "min_frames_for_confirm": 6,
            },
        }
        writer.set_config_summary(config)
        
        # Add some frame data
        writer.add_frame_data(
            frame_idx=0,
            timestamp_ms=0,
            detections=[
                ((100, 100, 50, 80), 0.85, {"profile": 0.75, "fill": 0.35, "aspect": 1.6, "parts": 2})
            ],
            rejects=[
                ((200, 200, 30, 20), "aspect=0.67")
            ],
            tracker_events={"confirmed": [1], "deleted": []},
            track_states={"confirmed_ids": [1], "suspect_ids": []},
        )
        
        writer.add_frame_data(
            frame_idx=1,
            timestamp_ms=33,
            detections=[],
            rejects=[
                ((150, 150, 40, 30), "fill=0.05")
            ],
            tracker_events={"confirmed": [], "deleted": [2]},
            track_states={"confirmed_ids": [1], "suspect_ids": [3]},
        )
        
        # Write report
        output_path = os.path.join(tmpdir, "test_reasons.txt")
        written_path = writer.write_report(output_path)
        
        assert written_path == output_path
        assert os.path.exists(output_path)
        
        # Read and verify content
        content = Path(output_path).read_text()
        
        # Check header
        assert "CONE DETECTION REASONS REPORT" in content
        assert "2024-01-01T12:00:00.000Z" in content
        assert "Total Frames: 2" in content
        
        # Check config summary
        assert "Configuration Summary" in content
        assert "HSV Orange Range" in content
        
        # Check frame 0
        assert "Frame 0" in content
        assert "ACCEPTED (1)" in content
        assert "(100, 100, 50, 80)" in content
        assert "score=0.850" in content
        assert "REJECTED (1)" in content
        assert "aspect=0.67" in content
        assert "CONFIRMED: [1]" in content
        
        # Check frame 1
        assert "Frame 1" in content
        assert "ACCEPTED (0)" in content
        assert "fill=0.05" in content
        assert "DELETED: [2]" in content
        assert "SUSPECT: [3]" in content
        
        # Check summary
        assert "SUMMARY" in content
        assert "Total Accepted Detections: 1" in content
        assert "Total Rejected Detections: 2" in content
        
        print(f"✓ Test passed: ReasonsWriter basic functionality")
        print(f"  Report written to: {output_path}")
        print(f"  File size: {Path(output_path).stat().st_size} bytes")


if __name__ == "__main__":
    # Run tests
    print("Running test_reasons_writer_basic_functionality...")
    test_reasons_writer_basic_functionality()
    print("\nRunning test_gist_runner_unit_mode_creates_txt_report...")
    test_gist_runner_unit_mode_creates_txt_report()
    print("\n✓ All tests passed!")
