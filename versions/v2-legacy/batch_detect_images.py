#!/usr/bin/env python3
"""
Batch image detection and analysis script for cone detection.

Processes all images in a DATASET folder, generates annotated outputs with bounding boxes,
and produces detailed logs with rejection reasons, suspect reasons, and detection info.

Usage:
    python3 batch_detect_images.py [--dataset FOLDER] [--output OUTPUT_FOLDER]
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Append parent directory so we can import cone_tracker
sys.path.insert(0, str(Path(__file__).parent))

from cone_tracker.config import load_config
from cone_tracker.detector import ConeDetector
from cone_tracker.tracker import MultiConeTracker
from cone_tracker.visualizer import Visualizer
from cone_tracker.utils import ConeState


class BatchImageDetector:
    """Process images in batch mode with detailed rejection/suspect logging."""

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    MIN_BBOX_HEIGHT_FOR_DISTANCE = 10.0

    def __init__(self, config=None):
        self.config = config if config is not None else load_config()
        self.detector = ConeDetector(self.config)
        self.tracker = MultiConeTracker(self.config)
        self.vis = Visualizer(self.config)
        self.results = []

    def _focal_px(self, frame_w: int) -> float:
        """Calculate focal length from horizontal field of view."""
        hfov_deg = self.config["camera"].get("hfov_deg", 70.0)
        hfov_deg = max(10.0, min(170.0, hfov_deg))
        hfov_rad = math.radians(hfov_deg)
        return (frame_w / 2.0) / math.tan(hfov_rad / 2.0)

    def _format_rejection_reason(self, reason: str) -> str:
        """Format rejection reason for display."""
        return reason.strip()

    def process_image(self, image_path: str, output_dir: str) -> dict:
        """
        Process a single image and return detection results.
        
        Processes the image repeatedly for at least 1 second to accumulate
        tracker data and improve detection precision.

        Args:
            image_path: Path to input image
            output_dir: Directory to save output annotated image

        Returns:
            Dictionary with detection results, rejections, and suspects
        """
        import time
        
        filename = Path(image_path).name
        logger.info(f"Processing: {filename}")

        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Failed to read image: {image_path}")
            return {
                'filename': filename,
                'status': 'error',
                'error': 'Failed to read image',
                'detections': [],
                'rejects': [],
                'suspects': []
            }

        # Resize to processing resolution
        proc_width = self.config["camera"]["process_width"]
        proc_height = self.config["camera"]["process_height"]
        proc = cv2.resize(frame, (proc_width, proc_height))

        # Reset tracker for this image to avoid cross-contamination
        self.tracker = MultiConeTracker(self.config)
        
        # Process image repeatedly for 1+ second to accumulate tracker data
        start_time = time.time()
        elapsed = 0
        iteration = 0
        all_rejects = []
        
        logger.info(f"  Processing for precision (1s minimum)...")
        
        while elapsed < 1.0:
            # Run detection
            detections, mask, rejects = self.detector.detect(proc)
            
            # Accumulate rejections
            if rejects:
                all_rejects.extend(rejects)
            
            # Update tracker with detections
            self.tracker.update(detections)
            
            iteration += 1
            elapsed = time.time() - start_time
            
            # Small sleep to avoid busy loop
            if elapsed < 1.0:
                time.sleep(0.01)
        
        # Remove duplicate rejections (same bbox and reason)
        seen_rejects = set()
        unique_rejects = []
        for bbox, reason in all_rejects:
            key = (tuple(bbox), reason)
            if key not in seen_rejects:
                seen_rejects.add(key)
                unique_rejects.append((bbox, reason))
        
        all_rejects = unique_rejects
        logger.info(f"  Processed {iteration} iterations in {elapsed:.2f}s")

        # Collect confirmed and suspect tracks
        confirmed_tracks = self.tracker.confirmed_tracks()
        suspect_tracks = [t for t in self.tracker.tracks if t.state == ConeState.SUSPECT]

        # Build detailed output
        focal_px = self._focal_px(proc_width)
        center_x = proc_width / 2.0
        cone_height_m = self.config["debug"].get("cone_height_m", None)

        detections_info = []
        for track in confirmed_tracks:
            err_px = track.cx - center_x
            angle_deg = math.degrees(math.atan(err_px / focal_px))
            est_dist_m = None
            if cone_height_m is not None and track.h > self.MIN_BBOX_HEIGHT_FOR_DISTANCE:
                est_dist_m = (cone_height_m * focal_px) / track.h

            detections_info.append({
                'track_id': track.track_id,
                'state': 'CONFIRMED',
                'bbox': list(track.bbox()),
                'cx': float(track.cx),
                'cy': float(track.cy),
                'width': float(track.w),
                'height': float(track.h),
                'err_px': float(err_px),
                'err_deg': float(angle_deg),
                'est_dist_m': float(est_dist_m) if est_dist_m else None,
                'avg_score': float(track.avg_score()),
                'frames_seen': track.frames_seen
            })

        suspects_info = []
        for track in suspect_tracks:
            err_px = track.cx - center_x
            angle_deg = math.degrees(math.atan(err_px / focal_px))
            suspects_info.append({
                'track_id': track.track_id,
                'state': 'SUSPECT',
                'bbox': list(track.bbox()),
                'cx': float(track.cx),
                'cy': float(track.cy),
                'width': float(track.w),
                'height': float(track.h),
                'err_px': float(err_px),
                'err_deg': float(angle_deg),
                'avg_score': float(track.avg_score()),
                'frames_seen': track.frames_seen,
                'reason': 'Below confirmation threshold or lost updates'
            })

        rejects_info = []
        for bbox, reason in all_rejects:
            x1, y1, x2, y2 = bbox
            rejects_info.append({
                'bbox': list(bbox),
                'width': float(x2 - x1),
                'height': float(y2 - y1),
                'reason': self._format_rejection_reason(reason)
            })

        # Draw on image with all information
        out = self.vis.draw(proc.copy(), confirmed_tracks + suspect_tracks, rejects,
                            fps=0.0, config_reload_msg=None)

        # Add text overlay with stats
        overlay = out.copy()
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 255, 0)

        # Title
        cv2.putText(overlay, f"Image: {filename}", (10, y_offset), font, font_scale, color, thickness)
        y_offset += 25
        cv2.putText(overlay, f"Confirmed: {len(confirmed_tracks)} | Suspects: {len(suspect_tracks)} | Rejected: {len(rejects)}", 
                   (10, y_offset), font, font_scale, color, thickness)
        y_offset += 25

        # Blend overlay
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

        # Save output image
        base_name = Path(image_path).stem
        output_filename = f"out_{base_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, out)
        logger.info(f"  Saved: {output_filename}")

        result = {
            'filename': filename,
            'status': 'success',
            'detections': detections_info,
            'suspects': suspects_info,
            'rejects': rejects_info,
            'output_image': output_filename,
            'stats': {
                'confirmed_count': len(confirmed_tracks),
                'suspect_count': len(suspect_tracks),
                'reject_count': len(all_rejects),
                'iterations': iteration,
                'processing_time_s': elapsed
            }
        }

        self.results.append(result)
        return result

    def process_directory(self, dataset_dir: str, output_dir: str) -> None:
        """
        Process all images in a directory.

        Args:
            dataset_dir: Directory containing images
            output_dir: Directory to save results
        """
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)

        if not dataset_path.is_dir():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return

        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Find all image files
        image_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_files.extend(dataset_path.glob(f"*{ext}"))
            image_files.extend(dataset_path.glob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))

        if not image_files:
            logger.warning(f"No images found in {dataset_dir}")
            return

        logger.info(f"Found {len(image_files)} image(s) to process")
        logger.info("=" * 70)

        # Process each image
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"\n[{i}/{len(image_files)}] Processing {image_path.name}")
            self.process_image(str(image_path), str(output_path))

        # Generate summary report
        self._write_summary_report(output_path)

    def _write_summary_report(self, output_dir: Path) -> None:
        """Write a comprehensive JSON and TXT report of all results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON report
        json_report = {
            'timestamp': timestamp,
            'config': {
                'process_width': self.config["camera"]["process_width"],
                'process_height': self.config["camera"]["process_height"],
                'hfov_deg': self.config["camera"]["hfov_deg"],
                'cone_height_m': self.config["debug"].get("cone_height_m")
            },
            'summary': {
                'total_images': len(self.results),
                'successful': sum(1 for r in self.results if r['status'] == 'success'),
                'errors': sum(1 for r in self.results if r['status'] == 'error'),
                'total_confirmed': sum(r['stats']['confirmed_count'] for r in self.results if r['status'] == 'success'),
                'total_suspects': sum(r['stats']['suspect_count'] for r in self.results if r['status'] == 'success'),
                'total_rejected': sum(r['stats']['reject_count'] for r in self.results if r['status'] == 'success')
            },
            'results': self.results
        }

        json_path = output_dir / f"detection_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        logger.info(f"\nJSON Report: {json_path.name}")

        # TXT report
        txt_path = output_dir / f"detection_report_{timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CONE DETECTION BATCH ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Images: {json_report['summary']['total_images']}\n")
            f.write(f"Successful: {json_report['summary']['successful']}\n")
            f.write(f"Errors: {json_report['summary']['errors']}\n\n")

            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Confirmed Cones: {json_report['summary']['total_confirmed']}\n")
            f.write(f"Total Suspect Tracks: {json_report['summary']['total_suspects']}\n")
            f.write(f"Total Rejections: {json_report['summary']['total_rejected']}\n\n")

            f.write("DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for result in self.results:
                f.write(f"Image: {result['filename']}\n")
                f.write("-" * 80 + "\n")

                if result['status'] == 'error':
                    f.write(f"ERROR: {result.get('error', 'Unknown')}\n\n")
                    continue

                f.write(f"Output: {result['output_image']}\n")
                f.write(f"Confirmed: {result['stats']['confirmed_count']}, "
                       f"Suspects: {result['stats']['suspect_count']}, "
                       f"Rejected: {result['stats']['reject_count']}\n\n")

                # Confirmed detections
                if result['detections']:
                    f.write("CONFIRMED DETECTIONS:\n")
                    for det in result['detections']:
                        f.write(f"  ID {det['track_id']}: score={det['avg_score']:.2f}, "
                               f"pos=({det['cx']:.0f},{det['cy']:.0f}), "
                               f"size=({det['width']:.0f}x{det['height']:.0f})\n")
                        f.write(f"           error={det['err_deg']:+.2f}°, "
                               f"distance={det['est_dist_m']:.2f}m\n")
                    f.write("\n")

                # Suspects
                if result['suspects']:
                    f.write("SUSPECT TRACKS:\n")
                    for susp in result['suspects']:
                        f.write(f"  ID {susp['track_id']}: score={susp['avg_score']:.2f}, "
                               f"frames_seen={susp['frames_seen']}\n")
                        f.write(f"           Reason: {susp['reason']}\n")
                    f.write("\n")

                # Rejections
                if result['rejects']:
                    f.write("REJECTION REASONS:\n")
                    rejection_counts = {}
                    for rej in result['rejects']:
                        reason = rej['reason']
                        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    
                    for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
                        f.write(f"  {count}x: {reason}\n")
                    f.write("\n")

                f.write("\n")

        logger.info(f"TXT Report: {txt_path.name}")
        logger.info("=" * 70)
        logger.info("BATCH PROCESSING COMPLETE")


def main():
    parser = argparse.ArgumentParser(
        description='Batch process images for cone detection with detailed analysis'
    )
    parser.add_argument('--dataset', default='DATASET',
                       help='Input dataset directory (default: DATASET)')
    parser.add_argument('--output', default='BATCH_OUTPUT',
                       help='Output directory for results (default: BATCH_OUTPUT)')
    parser.add_argument('--config', default='cone_config.yaml',
                       help='Configuration file (default: cone_config.yaml)')

    args = parser.parse_args()

    # Load config
    try:
        from cone_tracker.config import load_config
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Process
    detector = BatchImageDetector(config)
    detector.process_directory(args.dataset, args.output)


if __name__ == '__main__':
    main()
