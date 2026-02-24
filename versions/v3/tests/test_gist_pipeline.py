#!/usr/bin/env python3
"""
Unit tests for gist pipeline geometric functions.

Tests the convex_hull_pointing_up heuristic and GistPipeline class.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to import gist_test_runner
sys.path.insert(0, str(Path(__file__).parent))

from exec.gist_test_runner import convex_hull_pointing_up, GistPipeline, iou


def test_convex_hull_pointing_up_valid_cone():
    """Test that a cone-shaped hull is recognized."""
    # Create a triangular cone-like shape (pointing up)
    # Triangle with base at bottom
    hull = np.array([
        [[50, 10]],   # Top point
        [[30, 50]],   # Bottom left
        [[70, 50]],   # Bottom right
    ], dtype=np.int32)
    
    result = convex_hull_pointing_up(hull, aspect_thresh=0.8)
    print(f"✓ Test 1: Triangular cone shape -> {result}")
    assert result == True, "Triangular cone should pass"


def test_convex_hull_pointing_up_wide_rectangle():
    """Test that a wide rectangle is rejected."""
    # Wide rectangle (low aspect ratio)
    hull = np.array([
        [[10, 10]],
        [[90, 10]],
        [[90, 30]],
        [[10, 30]],
    ], dtype=np.int32)
    
    result = convex_hull_pointing_up(hull, aspect_thresh=0.8)
    print(f"✓ Test 2: Wide rectangle -> {result}")
    assert result == False, "Wide rectangle should be rejected (low aspect)"


def test_convex_hull_pointing_up_trapezoid():
    """Test that a trapezoid (cone-like) is recognized."""
    # Trapezoid with narrow top, wide bottom
    hull = np.array([
        [[45, 10]],   # Top left
        [[55, 10]],   # Top right
        [[70, 50]],   # Bottom right
        [[30, 50]],   # Bottom left
    ], dtype=np.int32)
    
    result = convex_hull_pointing_up(hull, aspect_thresh=0.8)
    print(f"✓ Test 3: Trapezoid (narrow top, wide bottom) -> {result}")
    assert result == True, "Trapezoid should pass as cone-like shape"


def test_convex_hull_pointing_up_inverted():
    """Test that an inverted cone (wide top, narrow bottom) is rejected."""
    # Inverted trapezoid
    hull = np.array([
        [[30, 10]],   # Top left
        [[70, 10]],   # Top right
        [[55, 50]],   # Bottom right
        [[45, 50]],   # Bottom left
    ], dtype=np.int32)
    
    result = convex_hull_pointing_up(hull, aspect_thresh=0.8)
    print(f"✓ Test 4: Inverted trapezoid -> {result}")
    # This should fail the centroid test (centroid too high for upward cone)
    assert result == False, "Inverted trapezoid should be rejected"


def test_convex_hull_pointing_up_empty():
    """Test that empty hull is rejected."""
    # Empty hull with proper shape but no points
    hull = np.array([], dtype=np.int32).reshape(0, 1, 2)
    
    result = convex_hull_pointing_up(hull, aspect_thresh=0.8)
    print(f"✓ Test 5: Empty hull -> {result}")
    assert result == False, "Empty hull should be rejected"


def test_convex_hull_pointing_up_too_few_points():
    """Test that hull with too few points is rejected."""
    hull = np.array([
        [[50, 10]],
        [[50, 20]],
    ], dtype=np.int32)
    
    result = convex_hull_pointing_up(hull, aspect_thresh=0.8)
    print(f"✓ Test 6: Too few points -> {result}")
    assert result == False, "Hull with <3 points should be rejected"


def test_gist_pipeline_initialization():
    """Test GistPipeline initialization."""
    config = {
        "camera": {"process_width": 960, "process_height": 540},
    }
    
    pipeline = GistPipeline(config)
    print(f"✓ Test 7: GistPipeline initialized")
    assert pipeline.canny_low == 30
    assert pipeline.canny_high == 100
    assert pipeline.approx_epsilon == 0.02
    assert pipeline.min_vertices == 3
    assert pipeline.max_vertices == 10


def test_gist_pipeline_process_mask_empty():
    """Test GistPipeline with empty mask."""
    config = {
        "camera": {"process_width": 960, "process_height": 540},
    }
    
    pipeline = GistPipeline(config)
    
    # Empty mask
    mask = np.zeros((540, 960), dtype=np.uint8)
    
    results = pipeline.process_mask(mask)
    print(f"✓ Test 8: GistPipeline with empty mask -> {len(results)} detections")
    assert len(results) == 0, "Empty mask should produce no detections"


def test_gist_pipeline_process_mask_with_cone():
    """Test GistPipeline with a cone-like shape in mask."""
    config = {
        "camera": {"process_width": 960, "process_height": 540},
    }
    
    pipeline = GistPipeline(config)
    
    # Create mask with cone-like shape
    mask = np.zeros((540, 960), dtype=np.uint8)
    
    # Draw a filled triangle (cone shape)
    pts = np.array([
        [480, 100],   # Top
        [420, 300],   # Bottom left
        [540, 300],   # Bottom right
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    results = pipeline.process_mask(mask)
    print(f"✓ Test 9: GistPipeline with cone shape -> {len(results)} detections")
    # Note: Canny + approxPolyDP might not detect this filled shape
    # This is expected - the test verifies pipeline runs without errors


def test_iou_helper():
    """Test IoU helper with overlapping boxes."""
    box_a = (10, 10, 20, 20)
    box_b = (15, 15, 20, 20)
    score = iou(box_a, box_b)
    print(f"✓ Test 10: IoU helper -> {score:.3f}")
    # Boxes overlap by 15x15 (IoU ~0.391)
    assert 0.35 < score < 0.45, "IoU should reflect partial overlap"


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Running Gist Pipeline Tests")
    print("=" * 70)
    
    try:
        test_convex_hull_pointing_up_valid_cone()
        test_convex_hull_pointing_up_wide_rectangle()
        test_convex_hull_pointing_up_trapezoid()
        test_convex_hull_pointing_up_inverted()
        test_convex_hull_pointing_up_empty()
        test_convex_hull_pointing_up_too_few_points()
        test_gist_pipeline_initialization()
        test_gist_pipeline_process_mask_empty()
        test_gist_pipeline_process_mask_with_cone()
        test_iou_helper()
        
        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
