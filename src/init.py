"""
SiPM Simulator Package

A Python package for simulating Silicon Photomultiplier (SiPM) response
to optical photons from particle physics experiments.

Main Components:
- SiPMSimulator: Core simulation class
- Utility functions for data processing and visualization
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .simulator import SiPMSimulator
from .utils import load_config, setup_output_directory

__all__ = [
    "SiPMSimulator",
    "load_config", 
    "setup_output_directory"
]
