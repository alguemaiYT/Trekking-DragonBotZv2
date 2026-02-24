# Gist Test Runner - High Precision Geometric Pipeline

## Overview

The Gist Test Runner is an independent test script that integrates a high-precision geometric pipeline for cone detection. This pipeline adds advanced validation steps on top of the existing `cone_tracker` modules:

- **Canny edge detection** for finding contours
- **Polygon approximation** (approxPolyDP) for shape simplification
- **Convex hull** computation for geometric analysis
- **convex_hull_pointing_up** heuristic for cone-like shape validation

The script is designed to **complement** (not replace) the existing cone detection system, providing:
- **Validation mode**: Compare gist detections vs standard detector
- **Experimental mode**: Use gist detections for tracking (--use-gist-acceptance flag)
- **CSV logging**: Detailed comparison metrics for analysis

## Features

✅ **Reuses existing modules**: `ConeDetector.preprocess()`, `get_mask()`, `MultiConeTracker`, `Visualizer`  
✅ **Independent testing**: No modifications to main cone_tracker code  
✅ **Dual-mode operation**: Validation (default) or Experimental (--use-gist-acceptance)  
✅ **CSV export**: Frame-by-frame comparison data in `logs/gist_test/`  
✅ **Visualization**: Color-coded overlays (Magenta=Gist, Cyan=Detector, Green/Yellow=Tracks)  
✅ **Configurable**: Uses `cone_config.yaml` settings  
✅ **No upscaling**: Processing resolution never exceeds source video size  

## Installation

The gist test runner requires the same dependencies as the main cone_tracker:

```bash
pip install opencv-python numpy pyyaml
```

## Usage

### Basic Usage (Validation Mode)

Run gist pipeline alongside standard detector for comparison:

```bash
python3 tests/exec/gist_test_runner.py --video path/to/video.mp4
```

This will:
1. Process video using **both** standard detector and gist pipeline
2. Compare detections (IoU-based matching)
3. Use **detector** results for tracking (default behavior)
4. Save CSV with comparison metrics to `logs/gist_test/gist_test_YYYYMMDD_HHMMSS.csv`

### Experimental Mode

Use gist detections for tracking (instead of detector):

```bash
python3 tests/exec/gist_test_runner.py --video path/to/video.mp4 --use-gist-acceptance
```

⚠️ **Warning**: This is experimental! Gist pipeline may produce different results than the tuned detector.

### Configuration File

Specify a custom config file:

```bash
python3 tests/exec/gist_test_runner.py --config my_config.yaml --video path/to/video.mp4
```

### Show Visualization

Enable OpenCV windows (overrides `debug.show_windows` in config):

```bash
python3 tests/exec/gist_test_runner.py --video path/to/video.mp4 --show-windows
```

### Limit Frames

Process only first N frames (useful for quick testing):

```bash
python3 tests/exec/gist_test_runner.py --video path/to/video.mp4 --max-frames 300
```

### IoU Threshold

Control the IoU threshold used for matching:

```bash
python3 tests/exec/gist_test_runner.py --video path/to/video.mp4 --iou-threshold 0.4
```

### IoU Histogram CSV

Write per-frame IoU histogram bins (0.0-1.0 in 0.1 steps) to CSV:

```bash
python3 tests/exec/gist_test_runner.py --video path/to/video.mp4 --iou-hist-csv logs/gist_iou_hist.csv
```

### Reasons Logging (Default Enabled)

**NEW**: The gist test runner now generates detailed per-frame reasons reports by default!

By default, the runner will:
- Enable detailed rejection reasons logging (`--log-reasons`)
- Generate a human-readable `.txt` report with per-frame detection/tracking details (`--reasons-txt`)
- Save the report to the repository root as `reasons_{timestamp}.txt`

The report includes:
- **Header**: Run timestamp, configuration summary (HSV ranges, tracking timeouts)
- **Per-frame sections**: 
  - Detector: Accepted detections (bbox, score, geometric data) and Rejected candidates (bbox, reason)
  - Gist: Candidates with geometric validation scores
  - Tracker: Confirmed/deleted track IDs and track states (CONFIRMED/SUSPECT)
- **Footer**: Summary statistics (total frames, accepted/rejected counts)

#### Customize Reasons Report

Specify a custom path for the reasons report:

```bash
python3 tests/exec/gist_test_runner.py --video path/to/video.mp4 --reasons-txt-path ./logs/my_reasons.txt
```

#### Disable Reasons Logging

If you want to disable reasons logging and report generation:

```bash
python3 tests/exec/gist_test_runner.py --video path/to/video.mp4 --no-log-reasons --no-reasons-txt
```

Or configure via `cone_config.yaml`:

```yaml
debug:
  log_reasons: false
  reasons_txt_enabled: false
```

### Unit Mode

Run a CI-friendly short pass (forces max 5 frames and disables windows):

```bash
python3 tests/exec/gist_test_runner.py --video path/to/video.mp4 --unit
```

### Custom Output Directory

Specify CSV output directory:

```bash
python3 tests/exec/gist_test_runner.py --video path/to/video.mp4 --output-dir my_logs/
```

## Command-Line Options

```
usage: gist_test_runner.py [-h] [--video VIDEO] [--config CONFIG] 
                           [--use-gist-acceptance] [--output-dir OUTPUT_DIR]
                           [--show-windows] [--max-frames MAX_FRAMES]
                           [--iou-threshold IOU_THRESHOLD]
                           [--iou-hist-csv IOU_HIST_CSV]
                           [--unit]

options:
  --video VIDEO               Path to video file (overrides cone_config.yaml)
  --config CONFIG             Path to config file (default: cone_config.yaml)
  --use-gist-acceptance       EXPERIMENTAL: Use gist detections for tracking
  --output-dir OUTPUT_DIR     CSV output directory (default: logs/gist_test)
  --show-windows              Show visualization windows (overrides config)
  --max-frames MAX_FRAMES     Maximum frames to process (default: all)
  --iou-threshold IOU_THRESHOLD
                              IoU threshold for matching detections (default: 0.3)
  --iou-hist-csv IOU_HIST_CSV
                              Optional CSV path for per-frame IoU histogram output
  --unit                      Unit mode: limit to 5 frames and disable visualization windows
```

## Pipeline Details

### Gist Geometric Pipeline Steps

1. **Preprocessing** (reuses `ConeDetector.preprocess()`):
   - Gaussian blur
   - BGR → HSV conversion
   - CLAHE on V channel

2. **Masking** (reuses `ConeDetector.get_mask()`):
   - Dual HSV range thresholding
   - Optional Lab/RG/backprojection masks
   - Morphological operations (open + close)

3. **Gist Pipeline** (new):
   - **Canny edge detection** (threshold: 30-100)
   - **Find contours** (RETR_EXTERNAL)
   - **Polygon approximation** (epsilon = 2% of perimeter)
   - **Vertex filtering** (3-10 vertices only)
   - **Convex hull** computation
   - **convex_hull_pointing_up** heuristic:
     - Aspect ratio check (height/width ≥ 0.8)
     - Vertical distribution analysis (top vs bottom thirds)
     - Convergence ratio (top narrower than bottom)
     - Centroid position (lower half for upward cone)

4. **Scoring**:
   - Aspect score: prefer 1.5-3.0 aspect ratio
   - Convexity score: ratio of hull area to contour area
   - Vertex score: prefer 4-7 vertices

### Visualization Color Coding

- **Magenta (255, 0, 255)**: Gist detections
- **Cyan (255, 255, 0)**: Standard detector detections
- **Green (0, 255, 0)**: Confirmed tracks (from tracker)
- **Yellow**: Suspect tracks (if `debug.draw_suspects=true`)

## CSV Output Format

The CSV file contains frame-by-frame comparison data:

| Column | Description |
|--------|-------------|
| `frame_idx` | Frame number (0-indexed) |
| `timestamp_ms` | Wall clock timestamp in milliseconds |
| `gist_detections` | Number of gist detections |
| `detector_detections` | Number of standard detector detections |
| `matched_detections` | Number of detections matched by IoU >= threshold |
| `gist_only` | Gist detections not matched by detector |
| `detector_only` | Detector detections not matched by gist |
| `gist_bbox_*` | Bounding box of first gist detection (x, y, w, h) |
| `gist_score` | Score of first gist detection |
| `gist_vertices` | Number of vertices in first gist detection |
| `detector_bbox_*` | Bounding box of first detector detection |
| `detector_score` | Score of first detector detection |

**Note**: Only the first detection of each type is logged in detail. Use for comparative analysis.

## Example Workflow

### 1. Validation Run (Compare Pipelines)

```bash
# Run on test video
python3 tests/exec/gist_test_runner.py \
  --video videos/test_cones.mp4 \
  --show-windows \
  --max-frames 500

# Analyze CSV output
python3 -c "
import pandas as pd
df = pd.read_csv('logs/gist_test/gist_test_*.csv')
print('Average gist detections:', df['gist_detections'].mean())
print('Average detector detections:', df['detector_detections'].mean())
print('Match rate:', df['matched_detections'].sum() / df['gist_detections'].sum())
"
```

### 2. Experimental Run (Use Gist for Tracking)

```bash
# Test gist-based tracking
python3 tests/exec/gist_test_runner.py \
  --video videos/test_cones.mp4 \
  --use-gist-acceptance \
  --show-windows
```

Compare tracking quality between validation and experimental modes.

## Testing

Unit tests for the gist pipeline are in `tests/test_gist_pipeline.py`:

```bash
python3 tests/test_gist_pipeline.py
```

Tests cover:
- ✅ `convex_hull_pointing_up()` heuristic (various shapes)
- ✅ `GistPipeline` initialization
- ✅ Mask processing (empty and filled shapes)

All tests should pass before using the gist runner.

## Integration Points

The gist test runner integrates with existing modules:

| Module | Usage |
|--------|-------|
| `cone_tracker.config` | `load_config()` for configuration |
| `cone_tracker.detector` | `ConeDetector.preprocess()`, `get_mask()`, `detect()` |
| `cone_tracker.tracker` | `MultiConeTracker.update()` for tracking |
| `cone_tracker.visualizer` | `Visualizer.draw()` for visualization |

**No modifications** are made to these modules - the gist runner is read-only.

## Troubleshooting

### No video specified error

```
ERROR: No video path specified. Use --video or set camera.video_path in config.
```

**Solution**: Provide video path via `--video` flag or set `camera.video_path` in `cone_config.yaml`.

### Video file not found

```
RuntimeError: Video file not found: path/to/video.mp4
```

**Solution**: Check file path is correct and file exists.

### No detections

If gist pipeline produces 0 detections on valid cones:
- Check mask quality: Are cones visible in HSV mask?
- Adjust Canny thresholds (edit `gist_test_runner.py`: `self.canny_low`, `self.canny_high`)
- Lower `aspect_thresh` parameter if cones are wider than expected
- Check vertex filtering (3-10 vertices) - complex shapes may be filtered out

### Too many false positives

If gist pipeline detects non-cone objects:
- Increase `aspect_thresh` (default: 0.8) to require taller shapes
- Tighten vertex filter range (e.g., 4-8 instead of 3-10)
- Increase minimum area threshold (currently 100px²)

## Limitations

- **Performance**: Gist pipeline is slower than standard detector (Canny + contour processing)
- **Edge-based**: Works best on frames with clear edges; may struggle with blurry or low-contrast cones
- **Parameter sensitivity**: `convex_hull_pointing_up` heuristic has hardcoded thresholds that may need tuning for different cone types
- **Single object focus**: CSV only logs first detection of each type in detail

## Future Enhancements

Potential improvements (not yet implemented):

- [ ] Configurable Canny/approx parameters via config file
- [ ] Adaptive thresholds based on frame content
- [ ] Multi-object CSV logging (currently only first detection)
- [ ] Performance profiling (time per pipeline stage)
- [ ] Hybrid mode: Use gist for high-confidence detections, detector for others
- [ ] Machine learning scoring (replace hand-crafted heuristic)

## References

- **Original gist**: High-precision geometric pipeline with convex_hull_pointing_up heuristic
- **Repository modules**: `cone_tracker/detector.py`, `cone_tracker/tracker.py`, `cone_tracker/visualizer.py`
- **OpenCV docs**: [Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html), [approxPolyDP](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c), [convexHull](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656)

## License

Same as parent repository (Trekking-DragonBotZv2).

---

**Author**: Copilot  
**Created**: 2026-01-20  
**Version**: 1.0
