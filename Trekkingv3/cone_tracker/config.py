#!/usr/bin/env python3
"""Configuration management for cone detection system."""
import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "video_path": "",  # Caminho para vídeo (deixe vazio para usar câmera)
        "output_video_path": "",  # Caminho para salvar vídeo processado (deixe vazio para não salvar)
        "index": 0,
        "capture_width": 1920,
        "capture_height": 1080,
        "process_width": 960,
        "process_height": 540,
        "fps": 30,
        "backend": "v4l2",
        "playback_mode": "fast",  # fast or realtime
        "max_consecutive_read_failures": 120,
        "hfov_deg": 70.0,  # Horizontal field of view in degrees
    },
    "debug": {
        "show_windows": True,
        "show_mask": True,
        "show_rejection_reason": True,  # Enable detailed rejection reasons by default
        "draw_suspects": False,     # if True, draws SUSPECT in yellow as well
        "show_groups": False,
        "log_rejections": False,    # Log rejections to console
        "log_suspects": False,      # Log suspects to console
        "show_heading_overlay": False,  # Show heading direction overlay on CONFIRMED tracks
        "heading_center_deadband_deg": 0.5,  # Deadband for CENTER classification in degrees
        "annotate_source_in_overlay": False,
        "export_run_log": False,
        "run_log_dir": "logs",
        "run_log_filename_pattern": "run_{source}_{start_ts}.jsonl",
        "structured_logging": False,  # Enable JSON logs when true
        "csv_export": {
            "enabled": False,
            "csv_dir": "logs/csv",
            "output_dir": "logs/csv",  # backward-compatible alias for csv_dir (csv_dir takes precedence)
            "filename_pattern": "run_{source}_{start_ts}.csv",
            "flush_every_frames": 10,
        },
        # Reasons logging and detailed .txt report generation
        "log_reasons": True,  # Enable detailed reasons logging (confirmations/rejections/suspects)
        "reasons_txt_enabled": True,  # Generate detailed .txt report by default
        "reasons_txt_path": "./reasons_{timestamp}.txt",  # Default path for reasons report (repo root)
    },
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
    "grouping": {
        "min_part_area": 80,
        "max_y_gap": 80,
        "min_x_overlap_ratio": 0.20,
        "max_x_center_diff": 80,
        "pad_x": 8,
        "pad_y": 12,
    },
    "geometry": {
        "min_group_area": 1400,  # px² ~ cone a 10m de distância na resolução de processamento
        "max_group_area": 450000,
        "aspect_min": 1.0,
        "aspect_max": 6.0,
        "profile_slices": 10,
        "min_profile_score": 0.35,
        "min_fill_ratio": 0.08,
        "max_fill_ratio": 0.65,
        "min_frame_score": 0.35,
        "confirm_avg_score": 0.55,
    },
    "weights": {
        "profile": 0.50,
        "fill": 0.35,
        "aspect": 0.15,
    },
    "tracking": {
        "max_tracks": 8,              # MULTI: maximum simultaneous cones
        "association_max_distance": 140,  # px ~ 15% da diagonal em 960x540, ajustar baseado na velocidade máxima do robô entre frames

        "ema_alpha": 0.25,
        "lost_timeout": 0.6,  # segundos, baseado na velocidade máxima de travessia do FoV
        "score_window": 10,
        "min_frames_for_confirm": 6,
        "grace_frames": 12,
        "grace_seconds": 0.0,
        "use_hungarian": False,
        "iou_penalty_weight": 0.35,
        "inertia_decay": 0.9,
        "min_confirmed_age_frames": 0,  # optional: require minimum age to display
    },
    "detector": {
        "type": "color",
        "params": {},
    },
    "clahe": {"clip_limit": 1.8, "tile_grid_size": [8, 8]},
    # Color/robustness options
    "color": {
        "enable_gray_world": True,
        "enable_lab_fallback": True,
        "lab_a_range": [140, 200],   # adjust if needed
        "lab_b_range": [130, 200],   # optional
        "enable_rg": False,
        "rg_thresholds": {"r_min": 0.30, "r_max": 0.75, "g_min": 0.12, "g_max": 0.50},
        "enable_backproj": False,
        "backproj_hist_path": "cone_hist.npy",
        "backproj_thresh": 50
    }
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Complexity: O(n) in total number of keys.
    """
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = "cone_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file, merging with defaults.

    Errors: logs and falls back to defaults on parse errors.
    """
    cfg = dict(DEFAULT_CONFIG)
    if not os.path.exists(path):
        return cfg
    try:
        with open(path, "r") as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
    except Exception as e:
        logger.exception(f"Failed to load config '{path}': {e}. Using DEFAULT_CONFIG.")
        cfg = dict(DEFAULT_CONFIG)
    cfg = deep_merge(DEFAULT_CONFIG, cfg)
    return cfg


def save_config(config: Dict[str, Any], path: str = "cone_config.yaml") -> None:
    """Save configuration to YAML file.

    Errors: logs exceptions on write failures.
    """
    try:
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config saved to {path}")
    except Exception as e:
        logger.exception(f"Failed to save config '{path}': {e}")


# Track last modification time for hot-reload
_config_mtime: Dict[str, float] = {}


def watch_config(path: str = "cone_config.yaml") -> bool:
    """
    Check if config file has been modified since last check.
    
    Args:
        path: Path to config file
        
    Returns:
        True if file has been modified or is being watched for first time
    """
    try:
        if not os.path.exists(path):
            return False
        
        current_mtime = os.path.getmtime(path)
        
        # First time watching this file
        if path not in _config_mtime:
            _config_mtime[path] = current_mtime
            return False
        
        # Check if modified
        if current_mtime > _config_mtime[path]:
            _config_mtime[path] = current_mtime
            return True
        
        return False
    except Exception as e:
        logger.exception(f"Failed to watch config '{path}': {e}")
        return False
