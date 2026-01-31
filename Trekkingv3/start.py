#!/usr/bin/env python3
"""
Main entry point for cone detection and tracking system.
This script uses the modular cone_tracker package.
"""
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from cone_tracker import App

if __name__ == "__main__":
    App().run()
