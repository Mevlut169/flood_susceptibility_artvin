"""
Flood Susceptibility Mapping Source Package
============================================
MYZ 305E - GeoAI Applications
Istanbul Technical University, Fall 2025

This package contains modules for:
- Data downloading and acquisition
- Preprocessing and alignment
- Feature engineering (terrain derivatives)
- Machine learning model training
- Visualization and mapping

Authors: Mevlütcan Yıldızlı, Uğur İnce
"""

__version__ = "1.0.0"
__author__ = "Mevlütcan Yıldızlı, Uğur İnce"
__email__ = "yildizli21@itu.edu.tr, inceu21@itu.edu.tr"

from .data_download import DataDownloader
from .preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model import FloodSusceptibilityModel
from .visualization import Visualizer
from .utils import setup_logging, timer, validate_raster

__all__ = [
    "DataDownloader",
    "DataPreprocessor", 
    "FeatureEngineer",
    "FloodSusceptibilityModel",
    "Visualizer",
    "setup_logging",
    "timer",
    "validate_raster"
]
