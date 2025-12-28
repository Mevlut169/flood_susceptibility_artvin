"""
Configuration Parameters for Flood Susceptibility Mapping
==========================================================
MYZ 305E - GeoAI Applications
Istanbul Technical University, Fall 2025

Authors: Mevlütcan Yıldızlı, Uğur İnce
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
MAPS_DIR = OUTPUT_DIR / "maps"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MAPS_DIR, FIGURES_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STUDY AREA CONFIGURATION - ARTVIN PROVINCE
# =============================================================================

STUDY_AREA = {
    "name": "Artvin Province",
    "country": "Türkiye",
    "bbox": {
        "min_lon": 41.0,
        "max_lon": 42.5,
        "min_lat": 40.7,
        "max_lat": 41.7
    },
    "area_km2": 7436,
    "epsg": 32637,  # UTM Zone 37N
    "epsg_name": "WGS 84 / UTM zone 37N"
}

# Bounding box as tuple (minx, miny, maxx, maxy)
BBOX = (
    STUDY_AREA["bbox"]["min_lon"],
    STUDY_AREA["bbox"]["min_lat"],
    STUDY_AREA["bbox"]["max_lon"],
    STUDY_AREA["bbox"]["max_lat"]
)

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

DATA_SOURCES = {
    "dem": {
        "name": "Copernicus DEM GLO-30",
        "url": "https://portal.opentopography.org/API/globaldem",
        "resolution": 30,  # meters
        "unit": "meters",
        "description": "Global 30m DEM from Copernicus Programme"
    },
    "rainfall": {
        "name": "CHIRPS v2.0",
        "url": "https://data.chc.ucsb.edu/products/CHIRPS-2.0/",
        "resolution": 5566,  # meters (~0.05 degrees)
        "unit": "mm/year",
        "years": range(2000, 2024),
        "description": "Climate Hazards Group InfraRed Precipitation with Station"
    },
    "landcover": {
        "name": "ESA WorldCover 2021",
        "url": "https://esa-worldcover.org/",
        "resolution": 10,  # meters
        "classes": {
            10: "Tree cover",
            20: "Shrubland",
            30: "Grassland",
            40: "Cropland",
            50: "Built-up",
            60: "Bare/sparse vegetation",
            70: "Snow and ice",
            80: "Permanent water bodies",
            90: "Herbaceous wetland",
            95: "Mangroves",
            100: "Moss and lichen"
        },
        "description": "10m global land cover map"
    },
    "rivers": {
        "name": "HydroRIVERS",
        "url": "https://www.hydrosheds.org/products/hydrorivers",
        "format": "shapefile",
        "description": "Global river network database"
    },
    "floods": {
        "name": "Global Flood Database",
        "url": "https://global-flood-database.cloudtostreet.ai/",
        "years": range(2000, 2019),
        "description": "Satellite-derived flood extent database"
    }
}

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================

PROCESSING = {
    "target_resolution": 30,  # meters
    "target_crs": f"EPSG:{STUDY_AREA['epsg']}",
    "nodata_value": -9999,
    "resampling_method": "bilinear",  # For continuous data
    "resampling_categorical": "nearest"  # For categorical data
}

# =============================================================================
# CONDITIONING FACTORS
# =============================================================================

CONDITIONING_FACTORS = {
    "elevation": {
        "name": "Elevation",
        "unit": "m",
        "source": "DEM",
        "type": "continuous",
        "flood_relation": "Lower areas accumulate water"
    },
    "slope": {
        "name": "Slope",
        "unit": "degrees",
        "source": "DEM derivative",
        "type": "continuous",
        "flood_relation": "Low slope causes water ponding"
    },
    "aspect": {
        "name": "Aspect",
        "unit": "degrees",
        "source": "DEM derivative",
        "type": "circular",
        "flood_relation": "Affects solar radiation and moisture"
    },
    "curvature": {
        "name": "Curvature",
        "unit": "1/m",
        "source": "DEM derivative",
        "type": "continuous",
        "flood_relation": "Concave areas collect water"
    },
    "twi": {
        "name": "Topographic Wetness Index",
        "unit": "dimensionless",
        "source": "Computed",
        "type": "continuous",
        "formula": "ln(a / tan(β))",
        "flood_relation": "Higher TWI = wetter conditions"
    },
    "spi": {
        "name": "Stream Power Index",
        "unit": "dimensionless",
        "source": "Computed",
        "type": "continuous",
        "formula": "a × tan(β)",
        "flood_relation": "Higher SPI = more erosive power"
    },
    "dist_river": {
        "name": "Distance to Rivers",
        "unit": "m",
        "source": "HydroRIVERS",
        "type": "continuous",
        "flood_relation": "Closer to rivers = higher exposure"
    },
    "drainage_density": {
        "name": "Drainage Density",
        "unit": "km/km²",
        "source": "Computed",
        "type": "continuous",
        "flood_relation": "Higher density = more flood paths"
    },
    "landcover": {
        "name": "Land Cover",
        "unit": "class",
        "source": "WorldCover",
        "type": "categorical",
        "flood_relation": "Impervious surfaces increase runoff"
    },
    "rainfall": {
        "name": "Annual Rainfall",
        "unit": "mm/year",
        "source": "CHIRPS",
        "type": "continuous",
        "flood_relation": "Higher rainfall = higher flood trigger"
    }
}

# =============================================================================
# RANDOM FOREST MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    "algorithm": "RandomForest",
    "params": {
        "n_estimators": 300,
        "max_depth": 20,
        "max_features": "sqrt",
        "min_samples_split": 5,
        "min_samples_leaf": 3,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": 1
    },
    "cv_folds": 5,
    "test_size": 0.3,
    "stratify": True
}

# Hyperparameter grid for tuning (optional)
PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", "log2", None]
}

# =============================================================================
# SAMPLING CONFIGURATION
# =============================================================================

SAMPLING = {
    "n_positive_samples": 847,  # From flood database
    "n_negative_samples": 847,  # Equal to positive (balanced)
    "min_separation_distance": 500,  # meters, to reduce autocorrelation
    "buffer_from_boundary": 100,  # meters, avoid edge effects
    "random_state": 42
}

# =============================================================================
# SUSCEPTIBILITY CLASSIFICATION
# =============================================================================

SUSCEPTIBILITY_CLASSES = {
    "method": "natural_breaks",  # or "equal_interval", "quantile"
    "n_classes": 5,
    "class_names": ["Very Low", "Low", "Moderate", "High", "Very High"],
    "colors": ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"],
    "thresholds": None  # Will be computed automatically
}

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

VISUALIZATION = {
    "figure_dpi": 300,
    "figure_format": "png",
    "map_style": "terrain-background",
    "colormap": "RdYlGn_r",
    "font_family": "Arial",
    "title_fontsize": 14,
    "label_fontsize": 12,
    "tick_fontsize": 10
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": OUTPUT_DIR / "flood_susceptibility.log"
}

# =============================================================================
# EXPECTED RESULTS (for validation)
# =============================================================================

EXPECTED_RESULTS = {
    "auc_min": 0.85,
    "auc_max": 0.95,
    "f1_min": 0.80,
    "high_risk_area_percent": 23.0,  # Approximately
    "top_features": ["slope", "dist_river", "twi", "elevation", "spi"]
}

# =============================================================================
# PRINT CONFIGURATION SUMMARY
# =============================================================================

def print_config():
    """Print configuration summary."""
    print("=" * 60)
    print("FLOOD SUSCEPTIBILITY MAPPING - CONFIGURATION")
    print("=" * 60)
    print(f"\n📍 Study Area: {STUDY_AREA['name']}, {STUDY_AREA['country']}")
    print(f"   Bounding Box: {BBOX}")
    print(f"   CRS: EPSG:{STUDY_AREA['epsg']}")
    print(f"\n📊 Model: {MODEL_CONFIG['algorithm']}")
    print(f"   Trees: {MODEL_CONFIG['params']['n_estimators']}")
    print(f"   Max Depth: {MODEL_CONFIG['params']['max_depth']}")
    print(f"   CV Folds: {MODEL_CONFIG['cv_folds']}")
    print(f"\n📁 Directories:")
    print(f"   Data: {DATA_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
