#!/usr/bin/env python3
"""
Gist Test Runner - High Precision Geometric Cone Detection Pipeline

This script integrates the geometric pipeline from the gist/notebook
(dual HSV segmentation, morphology, Canny, approxPolyDP, convexHull,
and convex_hull_pointing_up heuristic) as a validation/filtering layer
on top of the existing cone_tracker modules.

The pipeline reuses ConeDetector.preprocess() and get_mask() from the
main repository and adds geometric validation steps as an experimental
high-precision filter.

Reference: Gist pipeline with convex_hull_pointing_up heuristic
"""
import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add parent directory to path to import cone_tracker modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cone_tracker.config import load_config
from cone_tracker.detector import ConeDetector
from cone_tracker.tracker import MultiConeTracker
from cone_tracker.visualizer import Visualizer
from cone_tracker.reasons_writer import ReasonsWriter
from cone_tracker.utils import ConeState

logger = logging.getLogger(__name__)
IOU_HIST_BINS = 10


def convex_hull_pointing_up(
    convex_hull: np.ndarray,
    aspect_thresh: float = 0.8
) -> bool:
    """
    Heuristic to determine if a convex hull is pointing upward (cone-shaped).
    
    This function implements the geometric validation from the gist pipeline
    to filter cone-like shapes based on their convex hull properties.
    
    Reference: Gist notebook geometric pipeline
    
    Args:
        convex_hull: Convex hull points (Nx1x2 or Nx2 array)
        aspect_thresh: Minimum aspect ratio (height/width) to consider (default: 0.8)
        
    Returns:
        True if the convex hull represents a cone-like shape pointing upward
        
    Algorithm:
        1. Compute bounding box of convex hull
        2. Check aspect ratio (height/width >= aspect_thresh)
        3. Analyze vertical distribution: top third should be narrower than bottom
        4. Check if shape converges toward the top (cone-like)
    """
    if convex_hull is None or len(convex_hull) < 3:
        return False
    
    # Normalize hull shape to Nx2
    if len(convex_hull.shape) == 3:
        hull_pts = convex_hull.reshape(-1, 2)
    else:
        hull_pts = convex_hull
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(hull_pts)
    
    if w <= 0 or h <= 0:
        return False
    
    # Check aspect ratio (cones are taller than wide)
    aspect = h / float(w)
    if aspect < aspect_thresh:
        return False
    
    # Analyze vertical distribution - divide into thirds
    third_h = h / 3.0
    
    # Count points in top third vs bottom third
    top_y_max = y + third_h
    bottom_y_min = y + 2 * third_h
    
    top_points = hull_pts[(hull_pts[:, 1] <= top_y_max)]
    bottom_points = hull_pts[(hull_pts[:, 1] >= bottom_y_min)]
    
    # Need at least 1 point in each region to measure spread
    if len(top_points) < 1 or len(bottom_points) < 1:
        return False
    
    # Measure horizontal spread in top vs bottom third
    if len(top_points) == 1:
        top_x_spread = 0.0  # Single point has no spread
    else:
        top_x_spread = np.max(top_points[:, 0]) - np.min(top_points[:, 0])
    
    if len(bottom_points) == 1:
        bottom_x_spread = 0.0
    else:
        bottom_x_spread = np.max(bottom_points[:, 0]) - np.min(bottom_points[:, 0])
    
    # Cone should be narrower at top than bottom
    # For a cone, top_x_spread should be significantly less than bottom_x_spread
    if bottom_x_spread > 5:  # Only check if bottom has significant width
        convergence_ratio = top_x_spread / float(bottom_x_spread)
        if convergence_ratio > 0.90:  # Too uniform, not cone-like
            return False
    
    # Additional check: centroid should be in lower half for upward-pointing cone
    M = cv2.moments(hull_pts)
    if M["m00"] > 0:
        cy = M["m01"] / M["m00"]
        mid_y = y + h / 2.0
        # Centroid should be below middle (cone is heavier at bottom)
        if cy < mid_y - h * 0.1:  # Allow 10% tolerance
            return False
    
    return True


class GistPipeline:
    """
    Geometric pipeline processor for high-precision cone validation.
    
    This class implements the gist pipeline steps:
    - Canny edge detection
    - Contour extraction with findContours
    - Polygon approximation with approxPolyDP
    - Convex hull computation
    - Geometric filtering (3-10 vertices, convex_hull_pointing_up)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Canny parameters
        self.canny_low = 30
        self.canny_high = 100
        
        # Polygon approximation epsilon (as fraction of perimeter)
        self.approx_epsilon = 0.02
        
        # Vertex count filter
        self.min_vertices = 3
        self.max_vertices = 10
        
        # Aspect threshold for pointing up heuristic
        self.aspect_thresh = 0.8
        
        logger.info("GistPipeline initialized with Canny=[%d, %d], epsilon=%.3f",
                   self.canny_low, self.canny_high, self.approx_epsilon)
    
    def process_mask(
        self,
        mask: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:
        """
        Process binary mask through gist geometric pipeline.
        
        Args:
            mask: Binary mask (uint8, 0 or 255) from detector.get_mask()
            
        Returns:
            List of (approx_poly, convex_hull, bbox) for valid cone candidates
        """
        results = []
        
        # Step 1: Canny edge detection on mask
        edges = cv2.Canny(mask, self.canny_low, self.canny_high)
        
        # Step 2: Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # Skip very small contours
            area = cv2.contourArea(cnt)
            if area < 100:  # Minimum area threshold
                continue
            
            # Step 3: Approximate polygon
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.approx_epsilon * perimeter, True)
            
            # Step 4: Filter by vertex count (3-10 vertices)
            num_vertices = len(approx)
            if num_vertices < self.min_vertices or num_vertices > self.max_vertices:
                continue
            
            # Step 5: Compute convex hull
            hull = cv2.convexHull(approx)
            
            # Step 6: Apply pointing up heuristic
            if not convex_hull_pointing_up(hull, self.aspect_thresh):
                continue
            
            # Get bounding box
            bbox = cv2.boundingRect(hull)
            
            results.append((approx, hull, bbox))
        
        return results
    
    def compute_gist_score(
        self,
        approx: np.ndarray,
        hull: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Compute a quality score for gist detection.
        
        This score combines geometric properties to rank detections.
        
        Args:
            approx: Approximated polygon
            hull: Convex hull
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Score in [0, 1] range
        """
        x, y, w, h = bbox
        
        if w <= 0 or h <= 0:
            return 0.0
        
        # Aspect score (prefer 1.5-3.0 aspect ratio for cones)
        aspect = h / float(w)
        if aspect < 1.0:
            aspect_score = 0.0
        elif aspect < 1.5:
            aspect_score = (aspect - 1.0) / 0.5
        elif aspect <= 3.0:
            aspect_score = 1.0
        else:
            aspect_score = max(0.0, 1.0 - (aspect - 3.0) / 3.0)
        
        # Convexity score (ratio of hull area to contour area)
        hull_area = cv2.contourArea(hull)
        approx_area = cv2.contourArea(approx)
        
        if hull_area > 0:
            convexity = approx_area / hull_area
        else:
            convexity = 0.0
        
        convexity_score = convexity  # Higher is better (more convex)
        
        # Vertex count score (prefer 4-7 vertices for cone-like shapes)
        num_vertices = len(approx)
        if 4 <= num_vertices <= 7:
            vertex_score = 1.0
        else:
            vertex_score = 0.5
        
        # Combined score
        score = 0.5 * aspect_score + 0.3 * convexity_score + 0.2 * vertex_score
        
        return float(np.clip(score, 0.0, 1.0))


def create_video_capture(video_path: str) -> cv2.VideoCapture:
    """
    Create video capture with error handling.
    
    Args:
        video_path: Path to video file
        
    Returns:
        cv2.VideoCapture object
        
    Raises:
        RuntimeError: If video cannot be opened
    """
    if not os.path.exists(video_path):
        raise RuntimeError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    return cap


def setup_csv_output(output_dir: str) -> Tuple[str, csv.DictWriter, Any]:
    """
    Setup CSV output file for gist test results.
    
    Args:
        output_dir: Directory to save CSV (will be created if needed)
        
    Returns:
        Tuple of (csv_path, csv_writer, csv_file)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"gist_test_{timestamp}.csv")
    
    fieldnames = [
        "frame_idx",
        "timestamp_ms",
        "gist_detections",
        "detector_detections",
        "matched_detections",
        "gist_only",
        "detector_only",
        "gist_bbox_x",
        "gist_bbox_y",
        "gist_bbox_w",
        "gist_bbox_h",
        "gist_score",
        "gist_vertices",
        "detector_bbox_x",
        "detector_bbox_y",
        "detector_bbox_w",
        "detector_bbox_h",
        "detector_score",
    ]
    
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    logger.info("CSV output: %s", csv_path)
    
    return csv_path, writer, csv_file


def setup_iou_histogram_csv(output_path: str) -> Tuple[csv.writer, Any]:
    """Setup CSV output for per-frame IoU histogram."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    hist_file = None
    try:
        hist_file = open(output_path, "w", newline="")
        writer = csv.writer(hist_file)
        header = ["frame_idx"] + [
            f"bin_{i/IOU_HIST_BINS:.1f}_{(i + 1)/IOU_HIST_BINS:.1f}" for i in range(IOU_HIST_BINS)
        ]
        writer.writerow(header)
        logger.info("IoU histogram CSV output: %s", output_path)
        return writer, hist_file
    except Exception:
        if hist_file is not None:
            hist_file.close()
        raise


def compute_processing_resolution(
    config: Dict[str, Any],
    frame_w: int,
    frame_h: int
) -> Tuple[int, int, Optional[Tuple[int, int]]]:
    """Compute processing resolution without upscaling."""
    process_w = int(config["camera"]["process_width"])
    process_h = int(config["camera"]["process_height"])

    if frame_w > 0 and frame_h > 0:
        target_w = min(process_w, frame_w)
        scale = target_w / frame_w
        target_h = int(round(frame_h * scale))
        target_h = max(1, min(target_h, frame_h))
        return target_w, target_h, (frame_w, frame_h)

    return process_w, process_h, None


def clip_bbox(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    """Clip bounding box to frame bounds and normalize to ints."""
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return (0, 0, 0, 0)
    
    x, y, w, h = (int(round(v)) for v in bbox)
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))
    return (x, y, w, h)


def build_gist_feed_detections(
    gist_scored: List[Tuple[Tuple[int, int, int, int], float, Dict[str, Any]]],
    frame_shape: Tuple[int, int, int]
) -> List[Tuple[Tuple[int, int, int, int], float, Dict[str, Any]]]:
    """Build gist detections for tracker ingestion."""
    feed = []
    for bbox, score, _meta in gist_scored:
        clipped = clip_bbox(bbox, frame_shape)
        feed.append((clipped, float(score), {}))
    return feed


def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        boxA: First bbox (x, y, w, h)
        boxB: Second bbox (x, y, w, h)
        
    Returns:
        IoU score in [0, 1]
    """
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB
    
    # Convert to (x1, y1, x2, y2) format
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    # Compute intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    # Compute union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / float(union_area)


def main():
    """Main entry point for gist test runner."""
    parser = argparse.ArgumentParser(
        description="Gist Test Runner - High Precision Geometric Cone Detection Pipeline"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file (overrides cone_config.yaml camera.video_path)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cone_config.yaml",
        help="Path to configuration file (default: cone_config.yaml)"
    )
    parser.add_argument(
        "--use-gist-acceptance",
        action="store_true",
        help="EXPERIMENTAL: Use gist detections as candidates for tracking (instead of detector)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/gist_test",
        help="Directory for CSV output (default: logs/gist_test)"
    )
    parser.add_argument(
        "--show-windows",
        action="store_true",
        help="Show visualization windows (overrides config)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for matching detections (default: 0.3)"
    )
    parser.add_argument(
        "--iou-hist-csv",
        type=str,
        default=None,
        help="Optional CSV path for per-frame IoU histogram output"
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Unit mode: limit to 5 frames and disable visualization windows"
    )
    parser.add_argument(
        "--log-reasons",
        action="store_true",
        default=True,
        help="Enable detailed reasons logging (default: True)"
    )
    parser.add_argument(
        "--no-log-reasons",
        action="store_true",
        help="Disable reasons logging"
    )
    parser.add_argument(
        "--reasons-txt",
        action="store_true",
        default=True,
        help="Generate detailed .txt report (default: True)"
    )
    parser.add_argument(
        "--no-reasons-txt",
        action="store_true",
        help="Disable .txt report generation"
    )
    parser.add_argument(
        "--reasons-txt-path",
        type=str,
        default=None,
        help="Path for reasons .txt report (default: ./reasons_{timestamp}.txt)"
    )
    
    args = parser.parse_args()
    
    # Handle no-log-reasons and no-reasons-txt overrides
    if args.no_log_reasons:
        args.log_reasons = False
    if args.no_reasons_txt:
        args.reasons_txt = False
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    logger.info("=" * 70)
    logger.info("Gist Test Runner - High Precision Geometric Pipeline")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        logger.info("Loading configuration from: %s", args.config)
        config = load_config(args.config)
        
        # Override video path if provided
        video_path = args.video or config.get("camera", {}).get("video_path", "")
        
        if not video_path:
            logger.error("No video path specified. Use --video or set camera.video_path in config.")
            return 1
        
        # Override show_windows if requested
        if args.show_windows:
            config["debug"]["show_windows"] = True
        
        # Enable rejection reasons if log_reasons is enabled
        if args.log_reasons:
            config["debug"]["show_rejection_reason"] = True

        if args.unit:
            args.max_frames = 5
            config["debug"]["show_windows"] = False
            logger.info("Unit mode enabled: max_frames=5, show_windows disabled")
        
        # Initialize reasons writer if enabled
        reasons_writer = None
        if args.reasons_txt:
            reasons_writer = ReasonsWriter()
            reasons_writer.set_start_timestamp()
            reasons_writer.set_config_summary(config)
            logger.info("Reasons .txt report generation enabled")
        
        # Initialize components
        logger.info("Initializing ConeDetector, MultiConeTracker, Visualizer...")
        detector = ConeDetector(config)
        tracker = MultiConeTracker(config)
        visualizer = Visualizer(config)
        
        # Initialize gist pipeline
        logger.info("Initializing GistPipeline...")
        gist_pipeline = GistPipeline(config)
        
        # Setup video capture
        logger.info("Opening video: %s", video_path)
        cap = create_video_capture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_source = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info("Video properties: %dx%d @ %.1f fps, %d frames",
                   frame_w, frame_h, fps_source, total_frames)
        
        # Get processing resolution
        process_w, process_h, source_dims = compute_processing_resolution(
            config, frame_w, frame_h
        )
        if source_dims:
            logger.info("Processing resolution: %dx%d (source %dx%d)",
                        process_w, process_h, source_dims[0], source_dims[1])
        else:
            logger.info("Processing resolution: %dx%d (source unknown, using config)",
                        process_w, process_h)

        # Setup CSV output
        csv_path, csv_writer, csv_file = setup_csv_output(args.output_dir)
        iou_hist_writer = None
        iou_hist_file = None
        if args.iou_hist_csv:
            iou_hist_writer, iou_hist_file = setup_iou_histogram_csv(args.iou_hist_csv)
        
        # Processing loop
        frame_idx = 0
        start_time = time.time()
        fps_avg = 0.0
        
        logger.info("Starting processing loop...")
        logger.info("Mode: %s", 
                   "EXPERIMENTAL (gist acceptance)" if args.use_gist_acceptance else "VALIDATION (overlay only)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video or read failure at frame %d", frame_idx)
                    break
                
                # Check max frames limit
                if args.max_frames is not None and frame_idx >= args.max_frames:
                    logger.info("Reached max frames limit: %d", args.max_frames)
                    break
                
                # Resize frame to processing resolution
                if frame.shape[1] == process_w and frame.shape[0] == process_h:
                    frame_resized = frame
                else:
                    frame_resized = cv2.resize(frame, (process_w, process_h))
                
                # === Standard Detector Pipeline ===
                hsv = detector.preprocess(frame_resized)
                mask = detector.get_mask(frame_resized, hsv)
                
                # Run standard detector
                try:
                    detector_results, _, rejects = detector.detect(frame_resized)
                except Exception as exc:
                    logger.exception("Detector detect failed at frame %d: %s", frame_idx, exc)
                    detector_results = []
                    rejects = []
                
                # === Gist Pipeline on same mask ===
                gist_results = gist_pipeline.process_mask(mask)
                
                # Compute gist scores
                gist_scored = []
                for approx, hull, bbox in gist_results:
                    score = gist_pipeline.compute_gist_score(approx, hull, bbox)
                    gist_scored.append((bbox, score, {"approx": approx, "hull": hull, "vertices": len(approx)}))
                
                gist_feed = build_gist_feed_detections(gist_scored, frame_resized.shape)

                # === Decide which detections to track ===
                if args.use_gist_acceptance:
                    # EXPERIMENTAL: Use gist detections for tracking
                    tracking_input = gist_feed
                else:
                    # VALIDATION: Use detector detections for tracking
                    tracking_input = detector_results
                
                # Update tracker
                try:
                    tracker.update(tracking_input)
                except Exception as exc:
                    logger.exception("Tracker update failed at frame %d: %s", frame_idx, exc)
                
                # Collect reasons data for this frame if enabled
                if reasons_writer:
                    # Collect tracker events (newly confirmed, deleted)
                    tracker_events = {"confirmed": [], "deleted": []}
                    # Note: tracker doesn't expose confirmed/deleted events directly,
                    # we'd need to track state changes. For now, collect current states.
                    
                    # Collect track states
                    confirmed_ids = [t.track_id for t in tracker.tracks if t.state == ConeState.CONFIRMED]
                    suspect_ids = [t.track_id for t in tracker.tracks if t.state == ConeState.SUSPECT]
                    track_states = {
                        "confirmed_ids": confirmed_ids,
                        "suspect_ids": suspect_ids,
                    }
                    
                    # Prepare gist candidates for logging
                    gist_candidates_data = []
                    for bbox, score, meta in gist_scored:
                        gist_candidates_data.append({
                            "bbox": bbox,
                            "score": score,
                            "reason": f"geometric validation (vertices={meta['vertices']})"
                        })
                    
                    reasons_writer.add_frame_data(
                        frame_idx=frame_idx,
                        timestamp_ms=timestamp_ms,
                        detections=detector_results,
                        rejects=rejects,
                        gist_candidates=gist_candidates_data,
                        tracker_events=tracker_events,
                        track_states=track_states,
                    )
                
                # === Analysis: Compare gist vs detector ===
                matched = 0
                gist_only = 0
                detector_only = 0
                
                # Simple matching based on IoU
                matched_detector = set()
                matched_gist = set()
                
                for i, (gist_bbox, _, _) in enumerate(gist_feed):
                    best_iou = 0.0
                    best_j = -1
                    for j, (det_bbox, _, _) in enumerate(detector_results):
                        iou_score = iou(gist_bbox, det_bbox)
                        if iou_score > best_iou:
                            best_iou = iou_score
                            best_j = j
                    
                    if best_iou >= args.iou_threshold:
                        matched += 1
                        matched_detector.add(best_j)
                        matched_gist.add(i)
                
                gist_only = len(gist_feed) - len(matched_gist)
                detector_only = len(detector_results) - len(matched_detector)

                if iou_hist_writer is not None:
                    # Expected small detection counts; pairwise IoU is sufficient.
                    hist_counts = [0] * IOU_HIST_BINS
                    for gist_bbox, _, _ in gist_feed:
                        for det_bbox, _, _ in detector_results:
                            iou_score = iou(gist_bbox, det_bbox)
                            bin_idx = min(int(iou_score * IOU_HIST_BINS), IOU_HIST_BINS - 1)
                            hist_counts[bin_idx] += 1
                    iou_hist_writer.writerow([frame_idx] + hist_counts)
                
                # === Write CSV row ===
                timestamp_ms = int(time.time() * 1000)
                
                # Write summary row
                csv_writer.writerow({
                    "frame_idx": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "gist_detections": len(gist_feed),
                    "detector_detections": len(detector_results),
                    "matched_detections": matched,
                    "gist_only": gist_only,
                    "detector_only": detector_only,
                    "gist_bbox_x": gist_feed[0][0][0] if gist_feed else "",
                    "gist_bbox_y": gist_feed[0][0][1] if gist_feed else "",
                    "gist_bbox_w": gist_feed[0][0][2] if gist_feed else "",
                    "gist_bbox_h": gist_feed[0][0][3] if gist_feed else "",
                    "gist_score": f"{gist_scored[0][1]:.3f}" if gist_scored else "",
                    "gist_vertices": gist_scored[0][2]["vertices"] if gist_scored else "",
                    "detector_bbox_x": detector_results[0][0][0] if detector_results else "",
                    "detector_bbox_y": detector_results[0][0][1] if detector_results else "",
                    "detector_bbox_w": detector_results[0][0][2] if detector_results else "",
                    "detector_bbox_h": detector_results[0][0][3] if detector_results else "",
                    "detector_score": f"{detector_results[0][1]:.3f}" if detector_results else "",
                })
                
                # === Visualization ===
                # Draw gist detections in magenta
                for idx, (approx, hull, bbox) in enumerate(gist_results):
                    if idx < len(gist_feed):
                        x, y, w, h = gist_feed[idx][0]
                    else:
                        x, y, w, h = clip_bbox(bbox, frame_resized.shape)
                    cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    cv2.putText(frame_resized, "GIST", (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    # Draw convex hull
                    cv2.drawContours(frame_resized, [hull], 0, (255, 0, 255), 1)
                
                # Draw standard detector detections in cyan
                for (det_bbox, det_score, _) in detector_results:
                    x, y, w, h = det_bbox
                    cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(frame_resized, f"DET {det_score:.2f}", (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw tracks using visualizer
                try:
                    frame_vis = visualizer.draw(frame_resized, tracker.tracks, rejects, fps_avg)
                except Exception as exc:
                    logger.exception("Visualizer draw failed at frame %d: %s", frame_idx, exc)
                    frame_vis = frame_resized
                
                # Add info overlay
                cv2.putText(frame_vis, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_vis, f"Gist: {len(gist_feed)} | Detector: {len(detector_results)} | Matched: {matched}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                mode_text = "EXPERIMENTAL MODE" if args.use_gist_acceptance else "VALIDATION MODE"
                cv2.putText(frame_vis, mode_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show windows if enabled
                if config["debug"]["show_windows"]:
                    cv2.imshow("Gist Test Runner", frame_vis)
                    cv2.imshow("Mask", mask)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested quit")
                        break
                    elif key == ord(' '):
                        logger.info("Paused - press any key to continue")
                        cv2.waitKey(0)
                
                # Update FPS
                frame_idx += 1
                elapsed = time.time() - start_time
                fps_avg = frame_idx / elapsed if elapsed > 0 else 0.0
                
                # Progress log every 30 frames
                if frame_idx % 30 == 0:
                    gist_preview = [bbox for bbox, _, _ in gist_feed[:3]]
                    det_preview = [bbox for bbox, _, _ in detector_results[:3]]
                    logger.debug("Frame %d bbox preview - Gist: %s | Detector: %s",
                                 frame_idx, gist_preview, det_preview)
                    logger.info("Frame %d/%d (%.1f fps) - Gist: %d, Detector: %d, Matched: %d",
                               frame_idx, total_frames, fps_avg, len(gist_feed),
                               len(detector_results), matched)
        finally:
            # Ensure cleanup happens even on exception
            cap.release()
            csv_file.close()
            if iou_hist_file is not None:
                iou_hist_file.close()
            cv2.destroyAllWindows()
            
            # Write reasons report if enabled
            if reasons_writer:
                try:
                    reasons_txt_path = args.reasons_txt_path
                    if not reasons_txt_path:
                        # Auto-generate path in repo root
                        safe_ts = reasons_writer.start_timestamp.replace(":", "-").replace(".", "-")
                        reasons_txt_path = f"./reasons_{safe_ts}.txt"
                    
                    reasons_writer.write_report(
                        output_path=reasons_txt_path,
                        csv_path=csv_path,
                        iou_hist_path=args.iou_hist_csv,
                    )
                    logger.info("Reasons report written to: %s", reasons_txt_path)
                except Exception as exc:
                    logger.warning("Failed to write reasons report: %s", exc)
        
        # Final statistics
        logger.info("=" * 70)
        logger.info("Processing complete!")
        logger.info("Total frames processed: %d", frame_idx)
        logger.info("Average FPS: %.2f", fps_avg)
        logger.info("CSV output saved to: %s", csv_path)
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.exception("Fatal error in gist test runner: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
