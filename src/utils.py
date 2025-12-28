"""
Utility Functions for Flood Susceptibility Mapping
===================================================
MYZ 305E - GeoAI Applications

Contains helper functions for:
- Logging setup
- Timer decorators
- Raster validation
- Coordinate transformations
- File operations

Authors: Mevlütcan Yıldızlı, Uğur İnce
"""

import os
import sys
import time
import logging
import functools
from pathlib import Path
from typing import Tuple, Optional, Union, List
from datetime import datetime

import numpy as np

# Try to import geospatial libraries
try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_file : Path, optional
        Path to log file. If None, logs to console only.
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format_string : str, optional
        Custom format string for log messages
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    # Create logger
    logger = logging.getLogger("FloodSusceptibility")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Get default logger
logger = setup_logging()


# =============================================================================
# TIMER DECORATOR
# =============================================================================

def timer(func):
    """
    Decorator to measure and log function execution time.
    
    Usage
    -----
    @timer
    def my_function():
        pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting: {func.__name__}")
        
        result = func(*args, **kwargs)
        
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.2f} seconds"
        elif elapsed_time < 3600:
            time_str = f"{elapsed_time/60:.2f} minutes"
        else:
            time_str = f"{elapsed_time/3600:.2f} hours"
        
        logger.info(f"Completed: {func.__name__} in {time_str}")
        return result
    
    return wrapper


# =============================================================================
# RASTER UTILITIES
# =============================================================================

def validate_raster(filepath: Union[str, Path]) -> dict:
    """
    Validate a raster file and return its properties.
    
    Parameters
    ----------
    filepath : str or Path
        Path to raster file
        
    Returns
    -------
    dict
        Dictionary containing raster properties:
        - valid: bool
        - width, height: int
        - bands: int
        - crs: str
        - bounds: tuple
        - resolution: tuple
        - nodata: float
        - dtype: str
    """
    filepath = Path(filepath)
    result = {"valid": False, "path": str(filepath)}
    
    if not filepath.exists():
        result["error"] = "File does not exist"
        return result
    
    if HAS_RASTERIO:
        try:
            with rasterio.open(filepath) as src:
                result.update({
                    "valid": True,
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "crs": str(src.crs),
                    "bounds": src.bounds,
                    "resolution": src.res,
                    "nodata": src.nodata,
                    "dtype": str(src.dtypes[0]),
                    "transform": src.transform
                })
        except Exception as e:
            result["error"] = str(e)
    
    elif HAS_GDAL:
        try:
            ds = gdal.Open(str(filepath))
            if ds is None:
                result["error"] = "Failed to open with GDAL"
                return result
            
            gt = ds.GetGeoTransform()
            result.update({
                "valid": True,
                "width": ds.RasterXSize,
                "height": ds.RasterYSize,
                "bands": ds.RasterCount,
                "crs": ds.GetProjection(),
                "resolution": (abs(gt[1]), abs(gt[5])),
                "dtype": gdal.GetDataTypeName(ds.GetRasterBand(1).DataType),
                "nodata": ds.GetRasterBand(1).GetNoDataValue()
            })
            ds = None
        except Exception as e:
            result["error"] = str(e)
    else:
        result["error"] = "Neither rasterio nor GDAL available"
    
    return result


def get_raster_extent(filepath: Union[str, Path]) -> Tuple[float, float, float, float]:
    """
    Get raster extent as (minx, miny, maxx, maxy).
    
    Parameters
    ----------
    filepath : str or Path
        Path to raster file
        
    Returns
    -------
    tuple
        (minx, miny, maxx, maxy) bounds
    """
    info = validate_raster(filepath)
    if not info["valid"]:
        raise ValueError(f"Invalid raster: {info.get('error', 'Unknown error')}")
    
    if HAS_RASTERIO:
        with rasterio.open(filepath) as src:
            return (src.bounds.left, src.bounds.bottom, 
                    src.bounds.right, src.bounds.top)
    elif HAS_GDAL:
        ds = gdal.Open(str(filepath))
        gt = ds.GetGeoTransform()
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1] * ds.RasterXSize
        miny = maxy + gt[5] * ds.RasterYSize
        ds = None
        return (minx, miny, maxx, maxy)


def read_raster_as_array(
    filepath: Union[str, Path],
    band: int = 1,
    masked: bool = True
) -> np.ndarray:
    """
    Read raster band as numpy array.
    
    Parameters
    ----------
    filepath : str or Path
        Path to raster file
    band : int
        Band number (1-indexed)
    masked : bool
        If True, return masked array with nodata masked
        
    Returns
    -------
    np.ndarray
        Raster data as numpy array
    """
    if HAS_RASTERIO:
        with rasterio.open(filepath) as src:
            data = src.read(band)
            if masked and src.nodata is not None:
                data = np.ma.masked_equal(data, src.nodata)
            return data
    elif HAS_GDAL:
        ds = gdal.Open(str(filepath))
        rb = ds.GetRasterBand(band)
        data = rb.ReadAsArray()
        nodata = rb.GetNoDataValue()
        ds = None
        if masked and nodata is not None:
            data = np.ma.masked_equal(data, nodata)
        return data
    else:
        raise ImportError("Neither rasterio nor GDAL available")


# =============================================================================
# COORDINATE UTILITIES
# =============================================================================

def reproject_coords(
    x: float, y: float,
    src_epsg: int,
    dst_epsg: int
) -> Tuple[float, float]:
    """
    Reproject coordinates from one CRS to another.
    
    Parameters
    ----------
    x, y : float
        Input coordinates
    src_epsg : int
        Source EPSG code
    dst_epsg : int
        Destination EPSG code
        
    Returns
    -------
    tuple
        Reprojected (x, y) coordinates
    """
    if HAS_GDAL:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(src_epsg)
        
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(dst_epsg)
        
        transform = osr.CoordinateTransformation(src_srs, dst_srs)
        
        # GDAL 3+ uses (y, x) for geographic CRS
        if src_srs.IsGeographic():
            point = transform.TransformPoint(y, x)
        else:
            point = transform.TransformPoint(x, y)
        
        return (point[0], point[1])
    else:
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(
                f"EPSG:{src_epsg}", 
                f"EPSG:{dst_epsg}",
                always_xy=True
            )
            return transformer.transform(x, y)
        except ImportError:
            raise ImportError("Neither GDAL nor pyproj available for coordinate transformation")


def bbox_to_polygon(
    minx: float, miny: float, 
    maxx: float, maxy: float
) -> List[Tuple[float, float]]:
    """
    Convert bounding box to polygon coordinates.
    
    Parameters
    ----------
    minx, miny, maxx, maxy : float
        Bounding box coordinates
        
    Returns
    -------
    list
        List of (x, y) tuples forming closed polygon
    """
    return [
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny)  # Close polygon
    ]


# =============================================================================
# FILE UTILITIES
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Parameters
    ----------
    path : str or Path
        Directory path
        
    Returns
    -------
    Path
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(filepath: Union[str, Path]) -> str:
    """
    Get human-readable file size.
    
    Parameters
    ----------
    filepath : str or Path
        Path to file
        
    Returns
    -------
    str
        Human-readable file size (e.g., "1.5 MB")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return "N/A"
    
    size_bytes = filepath.stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} PB"


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory matching pattern.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search
    pattern : str
        Glob pattern (e.g., "*.tif")
    recursive : bool
        If True, search recursively
        
    Returns
    -------
    list
        List of Path objects
    """
    directory = Path(directory)
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


# =============================================================================
# DATA UTILITIES
# =============================================================================

def normalize_array(
    array: np.ndarray,
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0, 1)
) -> np.ndarray:
    """
    Normalize array values.
    
    Parameters
    ----------
    array : np.ndarray
        Input array
    method : str
        Normalization method: 'minmax', 'zscore', 'robust'
    feature_range : tuple
        Output range for minmax normalization
        
    Returns
    -------
    np.ndarray
        Normalized array
    """
    # Handle masked arrays
    if isinstance(array, np.ma.MaskedArray):
        mask = array.mask
        data = array.data.astype(float)
    else:
        mask = None
        data = array.astype(float)
    
    if method == "minmax":
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        if max_val - min_val > 0:
            normalized = (data - min_val) / (max_val - min_val)
            normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        else:
            normalized = np.zeros_like(data)
    
    elif method == "zscore":
        mean_val = np.nanmean(data)
        std_val = np.nanstd(data)
        if std_val > 0:
            normalized = (data - mean_val) / std_val
        else:
            normalized = np.zeros_like(data)
    
    elif method == "robust":
        median_val = np.nanmedian(data)
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1
        if iqr > 0:
            normalized = (data - median_val) / iqr
        else:
            normalized = np.zeros_like(data)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if mask is not None:
        normalized = np.ma.array(normalized, mask=mask)
    
    return normalized


def calculate_statistics(array: np.ndarray) -> dict:
    """
    Calculate basic statistics for array.
    
    Parameters
    ----------
    array : np.ndarray
        Input array
        
    Returns
    -------
    dict
        Dictionary with statistics
    """
    # Flatten and remove NaN/masked values
    if isinstance(array, np.ma.MaskedArray):
        data = array.compressed()
    else:
        data = array.flatten()
        data = data[~np.isnan(data)]
    
    return {
        "count": len(data),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "q1": float(np.percentile(data, 25)),
        "q3": float(np.percentile(data, 75))
    }


# =============================================================================
# TIMESTAMP UTILITIES
# =============================================================================

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}min"


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test logging
    logger = setup_logging()
    logger.info("Logging test successful")
    
    # Test timer
    @timer
    def test_func():
        time.sleep(0.1)
        return "done"
    
    result = test_func()
    print(f"Timer test: {result}")
    
    # Test normalize
    arr = np.array([1, 2, 3, 4, 5])
    norm = normalize_array(arr, method="minmax")
    print(f"Normalize test: {norm}")
    
    print("\nAll utility tests passed!")
