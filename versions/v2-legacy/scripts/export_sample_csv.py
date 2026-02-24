#!/usr/bin/env python3
"""Export debug CSV for a limited number of frames."""
import argparse
import os
import sys

from cone_tracker import App, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cone tracker and export debug CSV.")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to process")
    parser.add_argument("--output-dir", default=None, help="CSV output directory (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1

    config = load_config()
    config["camera"]["video_path"] = args.video
    config["camera"]["index"] = -1
    config["debug"]["show_windows"] = False
    config["debug"]["show_mask"] = False
    config["debug"].setdefault("csv_export", {})
    config["debug"]["csv_export"]["enabled"] = True
    if args.output_dir:
        config["debug"]["csv_export"]["output_dir"] = args.output_dir

    app = App(config=config)
    app.run(max_frames=args.frames)
    csv_path = app.csv_logger.csv_path if app.csv_logger else None
    if csv_path:
        print(csv_path)
        return 0
    print("CSV export was not created.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
