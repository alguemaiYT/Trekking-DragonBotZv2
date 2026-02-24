#!/bin/bash
# Quick setup and run script for batch cone detection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${1:-DATASET}"
OUTPUT_DIR="${2:-BATCH_OUTPUT}"

echo "=========================================="
echo "Cone Batch Detection - Setup & Run"
echo "=========================================="
echo ""
echo "Dataset directory: $DATASET_DIR"
echo "Output directory:  $OUTPUT_DIR"
echo ""

# Create DATASET if not exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Creating dataset directory: $DATASET_DIR"
    mkdir -p "$DATASET_DIR"
    echo "✓ Please copy your cone images into $DATASET_DIR/"
    echo ""
fi

# Check if DATASET has images
IMAGE_COUNT=$(find "$DATASET_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" \) | wc -l)

if [ $IMAGE_COUNT -eq 0 ]; then
    echo "⚠ No images found in $DATASET_DIR"
    echo "Supported formats: .png, .jpg, .jpeg, .bmp, .tiff, .webp"
    echo ""
    echo "Copy images and run again:"
    echo "  cp /path/to/images/* $DATASET_DIR/"
    echo "  $0"
    exit 0
fi

echo "Found $IMAGE_COUNT image(s) in $DATASET_DIR"
echo ""

# Create output dir
mkdir -p "$OUTPUT_DIR"
echo "Output will be saved to: $OUTPUT_DIR"
echo ""

# Run batch detection
echo "Starting batch detection..."
echo "=========================================="
cd "$SCRIPT_DIR"
python3 batch_detect_images.py --dataset "$DATASET_DIR" --output "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "✓ Processing complete!"
echo ""
echo "Check results:"
echo "  - Annotated images: $OUTPUT_DIR/out_*.png"
echo "  - JSON report:      $OUTPUT_DIR/detection_report_*.json"
echo "  - TXT report:       $OUTPUT_DIR/detection_report_*.txt"
echo ""
echo "View text report:"
echo "  cat $(ls -t $OUTPUT_DIR/detection_report_*.txt | head -1)"
echo ""
