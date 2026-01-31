#!/usr/bin/env python3
"""
Demonstration script for gist test runner with synthetic cone frames.

This script generates synthetic frames with cone-like shapes and runs
the gist pipeline to demonstrate functionality without requiring a real video file.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to import gist_test_runner
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cone_tracker.config import load_config
from cone_tracker.detector import ConeDetector
from tests.exec.gist_test_runner import GistPipeline, convex_hull_pointing_up


def create_synthetic_cone_frame(width=960, height=540, num_cones=2):
    """
    Create a synthetic frame with cone-like shapes.
    
    Args:
        width: Frame width
        height: Frame height
        num_cones: Number of cone shapes to draw
        
    Returns:
        BGR frame with synthetic cones
    """
    # Create white background
    frame = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Draw cone-like shapes (orange trapezoids)
    for i in range(num_cones):
        # Random position
        base_x = int(width * (i + 1) / (num_cones + 1))
        base_y = int(height * 0.7)
        
        # Cone dimensions
        base_width = 80
        top_width = 30
        cone_height = 150
        
        # Define trapezoid points (cone shape)
        pts = np.array([
            [base_x - top_width // 2, base_y - cone_height],  # Top left
            [base_x + top_width // 2, base_y - cone_height],  # Top right
            [base_x + base_width // 2, base_y],                # Bottom right
            [base_x - base_width // 2, base_y],                # Bottom left
        ], dtype=np.int32)
        
        # Fill with orange color (BGR: 0, 127, 255)
        cv2.fillPoly(frame, [pts], (0, 127, 255))
        
        # Add some white stripes for realism
        stripe_y = base_y - cone_height + 30
        stripe_pts = np.array([
            [base_x - top_width // 2 + 5, stripe_y],
            [base_x + top_width // 2 - 5, stripe_y],
            [base_x + top_width // 2 + 10, stripe_y + 20],
            [base_x - top_width // 2 - 10, stripe_y + 20],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [stripe_pts], (255, 255, 255))
    
    return frame


def demo_convex_hull_heuristic():
    """Demonstrate convex_hull_pointing_up heuristic."""
    print("=" * 70)
    print("Demo 1: convex_hull_pointing_up Heuristic")
    print("=" * 70)
    
    # Test 1: Cone-like shape
    cone_hull = np.array([
        [[50, 10]],   # Top
        [[30, 50]],   # Bottom left
        [[70, 50]],   # Bottom right
    ], dtype=np.int32)
    
    result = convex_hull_pointing_up(cone_hull)
    print(f"✓ Cone-like triangle: {result}")
    assert result == True
    
    # Test 2: Wide rectangle
    rect_hull = np.array([
        [[10, 10]],
        [[90, 10]],
        [[90, 30]],
        [[10, 30]],
    ], dtype=np.int32)
    
    result = convex_hull_pointing_up(rect_hull)
    print(f"✓ Wide rectangle: {result}")
    assert result == False
    
    print("✓ Heuristic working correctly!\n")


def demo_gist_pipeline_on_synthetic():
    """Demonstrate GistPipeline on synthetic cone frames."""
    print("=" * 70)
    print("Demo 2: GistPipeline on Synthetic Frames")
    print("=" * 70)
    
    # Load config with error handling
    try:
        config = load_config("cone_config.yaml")
    except Exception as e:
        print(f"⚠ Warning: Could not load cone_config.yaml ({e}), using defaults")
        # Minimal config for demo
        config = {
            "camera": {"process_width": 960, "process_height": 540},
            "debug": {"show_windows": False},
            "hsv_orange": {
                "low_1": [0, 90, 90],
                "high_1": [28, 255, 255],
                "low_2": [160, 90, 80],
                "high_2": [179, 255, 255],
            },
            "morphology": {
                "kernel_open": 3,
                "kernel_close": 7,
                "open_iterations": 1,
                "close_iterations": 1,
            },
            "clahe": {"clip_limit": 1.8, "tile_grid_size": [8, 8]},
            "color": {},
            "geometry": {
                "min_group_area": 1400,
                "max_group_area": 450000,
                "aspect_min": 1.0,
                "aspect_max": 6.0,
                "profile_slices": 10,
                "min_profile_score": 0.35,
                "min_fill_ratio": 0.08,
                "max_fill_ratio": 0.65,
                "min_frame_score": 0.35,
            },
            "grouping": {
                "min_part_area": 80,
                "max_y_gap": 80,
                "min_x_overlap_ratio": 0.20,
                "max_x_center_diff": 80,
                "pad_x": 8,
                "pad_y": 12,
            },
            "weights": {
                "profile": 0.50,
                "fill": 0.35,
                "aspect": 0.15,
            },
        }
    
    # Initialize detector and gist pipeline
    detector = ConeDetector(config)
    gist_pipeline = GistPipeline(config)
    
    # Create synthetic frame
    frame = create_synthetic_cone_frame(width=960, height=540, num_cones=2)
    print("✓ Synthetic frame created (960x540, 2 cones)")
    
    # Run detector pipeline
    hsv = detector.preprocess(frame)
    mask = detector.get_mask(frame, hsv)
    detector_results, _, _ = detector.detect(frame)
    
    print(f"✓ Detector found {len(detector_results)} detections")
    
    # Run gist pipeline
    gist_results = gist_pipeline.process_mask(mask)
    print(f"✓ Gist pipeline found {len(gist_results)} candidates")
    
    # Show results for each gist detection
    for i, (approx, hull, bbox) in enumerate(gist_results):
        score = gist_pipeline.compute_gist_score(approx, hull, bbox)
        x, y, w, h = bbox
        aspect = h / float(w) if w > 0 else 0
        print(f"  - Gist detection {i+1}: bbox=({x},{y},{w},{h}), "
              f"aspect={aspect:.2f}, score={score:.2f}, vertices={len(approx)}")
    
    # Visualize
    print("\n✓ Saving visualization to /tmp/gist_demo.png...")
    vis = frame.copy()
    
    # Draw gist detections in magenta
    for approx, hull, bbox in gist_results:
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.drawContours(vis, [hull], 0, (255, 0, 255), 1)
        cv2.putText(vis, "GIST", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Draw detector detections in cyan
    for (det_bbox, det_score, _) in detector_results:
        x, y, w, h = det_bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(vis, f"DET {det_score:.2f}", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Save visualization
    cv2.imwrite("/tmp/gist_demo.png", vis)
    print("✓ Visualization saved!\n")
    
    # Also save original frame and mask
    cv2.imwrite("/tmp/gist_demo_frame.png", frame)
    cv2.imwrite("/tmp/gist_demo_mask.png", mask)
    print("✓ Saved: /tmp/gist_demo_frame.png (original)")
    print("✓ Saved: /tmp/gist_demo_mask.png (HSV mask)")


def main():
    """Main demo entry point."""
    print("\n" + "=" * 70)
    print("Gist Test Runner - Synthetic Demonstration")
    print("=" * 70 + "\n")
    
    try:
        # Demo 1: Heuristic
        demo_convex_hull_heuristic()
        
        # Demo 2: Pipeline on synthetic frames
        demo_gist_pipeline_on_synthetic()
        
        print("=" * 70)
        print("✅ All demos completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Check visualizations in /tmp/gist_demo*.png")
        print("  2. Run with real video: python3 tests/exec/gist_test_runner.py --video path/to/video.mp4")
        print("  3. Read documentation: tests/exec/GIST_RUNNER_README.md")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
