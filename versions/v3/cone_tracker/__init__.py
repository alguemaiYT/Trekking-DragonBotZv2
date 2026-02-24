#!/usr/bin/env python3
"""Cone Tracker - A modular cone detection and tracking system."""
import logging

from .app import App
from .config import DEFAULT_CONFIG, load_config, save_config
from .detector import BaseDetector, ColorDetector, ConeDetector, DetectorFusion
from .tracker import MultiConeTracker, Track
from .utils import ConeState, VisibilityState
from .visualizer import Visualizer

__version__ = "1.0.0"
__all__ = [
    "App",
    "ConeDetector",
    "BaseDetector",
    "ColorDetector",
    "DetectorFusion",
    "MultiConeTracker",
    "Track",
    "ConeState",
    "VisibilityState",
    "Visualizer",
    "load_config",
    "save_config",
    "DEFAULT_CONFIG",
]

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
