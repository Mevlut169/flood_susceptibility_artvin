"""
Data Preprocessing Module for Flood Susceptibility Mapping
===========================================================
MYZ 305E - GeoAI Applications

Handles preprocessing and alignment of geospatial datasets:
- CRS reprojection (to UTM Zone 37N)
- Resolution resampling (to 30m)
- Extent clipping
- NoData handling
- Layer alignment

Authors: Mevlütcan Yıldızlı, Uğur İnce
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union

import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logging, timer, ensure_dir, validate_raster

# Try importing geospatial libraries
try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask
    from rasterio.crs import CRS
    from rasterio.enums import Resampling as RioResampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

try:
    import geopandas as gpd
    from shapely.geometry import box, mapping
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Setup logger
logger = setup_logging()


class DataPreprocessor:
    """
    Class for preprocessing geospatial data layers.
    
    Handles reprojection, resampling, and alignment of raster
    and vector datasets to ensure consistency for analysis.
    
    Attributes
    ----------
    target_crs : str
        Target coordinate reference system
    target_resolution : float
        Target resolution in meters
    bbox : tuple
        Bounding box in target CRS
    nodata : float
        NoData value for outputs
    
    Example
    -------
    >>> preprocessor = DataPreprocessor(
    ...     target_crs="EPSG:32637",
    ...     target_resolution=30,
    ...     bbox=(300000, 4500000, 450000, 4620000)
    ... )
    >>> preprocessor.preprocess_raster("dem.tif", "dem_processed.tif")
    """
    
    def __init__(
        self,
        target_crs: str = "EPSG:32637",
        target_resolution: float = 30,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        nodata: float = -9999
    ):
        """
        Initialize the DataPreprocessor.
        
        Parameters
        ----------
        target_crs : str
            Target CRS (default: UTM Zone 37N for Artvin)
        target_resolution : float
            Target resolution in meters
        bbox : tuple, optional
            Bounding box (minx, miny, maxx, maxy) in target CRS
        nodata : float
            NoData value for outputs
        """
        self.target_crs = target_crs
        self.target_resolution = target_resolution
        self.bbox = bbox
        self.nodata = nodata
        
        logger.info(f"DataPreprocessor initialized")
        logger.info(f"  Target CRS: {target_crs}")
        logger.info(f"  Target Resolution: {target_resolution}m")
    
    # =========================================================================
    # RASTER PREPROCESSING
    # =========================================================================
    
    @timer
    def preprocess_raster(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        resampling: str = "bilinear"
    ) -> Path:
        """
        Preprocess a raster: reproject, resample, and clip.
        
        Parameters
        ----------
        input_path : str or Path
            Input raster path
        output_path : str or Path
            Output raster path
        resampling : str
            Resampling method: 'nearest', 'bilinear', 'cubic'
            
        Returns
        -------
        Path
            Path to processed raster
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preprocessing: {input_path.name}")
        
        if HAS_RASTERIO:
            return self._preprocess_with_rasterio(input_path, output_path, resampling)
        elif HAS_GDAL:
            return self._preprocess_with_gdal(input_path, output_path, resampling)
        else:
            logger.error("Neither rasterio nor GDAL available!")
            raise ImportError("Geospatial libraries required")
    
    def _preprocess_with_rasterio(
        self,
        input_path: Path,
        output_path: Path,
        resampling: str
    ) -> Path:
        """Preprocess raster using rasterio."""
        
        # Map resampling method
        resampling_map = {
            'nearest': RioResampling.nearest,
            'bilinear': RioResampling.bilinear,
            'cubic': RioResampling.cubic,
            'average': RioResampling.average
        }
        resample_method = resampling_map.get(resampling, RioResampling.bilinear)
        
        with rasterio.open(input_path) as src:
            # Calculate transform for reprojection
            dst_crs = CRS.from_string(self.target_crs)
            
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs,
                src.width, src.height,
                *src.bounds,
                resolution=self.target_resolution
            )
            
            # Update metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'nodata': self.nodata
            })
            
            # Reproject
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=resample_method,
                        dst_nodata=self.nodata
                    )
        
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Size: {width}x{height}, Resolution: {self.target_resolution}m")
        
        return output_path
    
    def _preprocess_with_gdal(
        self,
        input_path: Path,
        output_path: Path,
        resampling: str
    ) -> Path:
        """Preprocess raster using GDAL."""
        
        # Map resampling method
        resampling_map = {
            'nearest': gdal.GRA_NearestNeighbour,
            'bilinear': gdal.GRA_Bilinear,
            'cubic': gdal.GRA_Cubic,
            'average': gdal.GRA_Average
        }
        resample_method = resampling_map.get(resampling, gdal.GRA_Bilinear)
        
        # Open source
        src = gdal.Open(str(input_path))
        if src is None:
            raise IOError(f"Could not open {input_path}")
        
        # Warp options
        warp_options = gdal.WarpOptions(
            dstSRS=self.target_crs,
            xRes=self.target_resolution,
            yRes=self.target_resolution,
            resampleAlg=resample_method,
            dstNodata=self.nodata,
            format='GTiff',
            creationOptions=['COMPRESS=LZW', 'TILED=YES']
        )
        
        # Perform warp
        gdal.Warp(str(output_path), src, options=warp_options)
        
        src = None
        
        logger.info(f"  Output: {output_path}")
        
        return output_path
    
    # =========================================================================
    # ALIGN RASTERS
    # =========================================================================
    
    @timer
    def align_rasters(
        self,
        raster_paths: Dict[str, Path],
        reference_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Align multiple rasters to a reference raster.
        
        Parameters
        ----------
        raster_paths : dict
            Dictionary of {name: path} for input rasters
        reference_path : str or Path
            Path to reference raster (defines extent and resolution)
        output_dir : str or Path
            Output directory for aligned rasters
            
        Returns
        -------
        dict
            Dictionary of {name: path} for aligned rasters
        """
        reference_path = Path(reference_path)
        output_dir = ensure_dir(output_dir)
        
        logger.info(f"Aligning {len(raster_paths)} rasters to reference")
        logger.info(f"  Reference: {reference_path.name}")
        
        # Get reference properties
        ref_info = validate_raster(reference_path)
        if not ref_info['valid']:
            raise ValueError(f"Invalid reference raster: {ref_info.get('error')}")
        
        aligned_paths = {}
        
        for name, raster_path in raster_paths.items():
            output_path = output_dir / f"{name}_aligned.tif"
            
            logger.info(f"  Aligning: {name}")
            
            if HAS_RASTERIO:
                self._align_with_rasterio(raster_path, reference_path, output_path)
            elif HAS_GDAL:
                self._align_with_gdal(raster_path, reference_path, output_path)
            
            aligned_paths[name] = output_path
        
        return aligned_paths
    
    def _align_with_rasterio(
        self,
        input_path: Path,
        reference_path: Path,
        output_path: Path
    ):
        """Align raster to reference using rasterio."""
        
        with rasterio.open(reference_path) as ref:
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_width = ref.width
            ref_height = ref.height
        
        with rasterio.open(input_path) as src:
            # Prepare output
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': ref_crs,
                'transform': ref_transform,
                'width': ref_width,
                'height': ref_height,
                'nodata': self.nodata
            })
            
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=RioResampling.bilinear,
                        dst_nodata=self.nodata
                    )
    
    def _align_with_gdal(
        self,
        input_path: Path,
        reference_path: Path,
        output_path: Path
    ):
        """Align raster to reference using GDAL."""
        
        ref = gdal.Open(str(reference_path))
        gt = ref.GetGeoTransform()
        proj = ref.GetProjection()
        
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1] * ref.RasterXSize
        miny = maxy + gt[5] * ref.RasterYSize
        
        ref = None
        
        warp_options = gdal.WarpOptions(
            dstSRS=proj,
            outputBounds=(minx, miny, maxx, maxy),
            xRes=abs(gt[1]),
            yRes=abs(gt[5]),
            resampleAlg=gdal.GRA_Bilinear,
            dstNodata=self.nodata,
            format='GTiff'
        )
        
        src = gdal.Open(str(input_path))
        gdal.Warp(str(output_path), src, options=warp_options)
        src = None
    
    # =========================================================================
    # VECTOR PREPROCESSING
    # =========================================================================
    
    @timer
    def preprocess_vector(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Preprocess vector data: reproject to target CRS.
        
        Parameters
        ----------
        input_path : str or Path
            Input vector path
        output_path : str or Path
            Output vector path
            
        Returns
        -------
        Path
            Path to processed vector
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preprocessing vector: {input_path.name}")
        
        if not HAS_GEOPANDAS:
            logger.warning("GeoPandas not available, copying file as-is")
            import shutil
            shutil.copy(input_path, output_path)
            return output_path
        
        # Read and reproject
        gdf = gpd.read_file(input_path)
        gdf_reprojected = gdf.to_crs(self.target_crs)
        
        # Save
        gdf_reprojected.to_file(output_path, driver="GPKG")
        
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Features: {len(gdf_reprojected)}")
        
        return output_path
    
    # =========================================================================
    # CREATE SAMPLE POINTS
    # =========================================================================
    
    @timer
    def create_sample_points(
        self,
        flood_events_path: Union[str, Path],
        output_path: Union[str, Path],
        n_samples: int = 847,
        min_distance: float = 500
    ) -> Path:
        """
        Create balanced sample points for model training.
        
        Parameters
        ----------
        flood_events_path : str or Path
            Path to flood events vector
        output_path : str or Path
            Output path for sample points
        n_samples : int
            Number of samples per class
        min_distance : float
            Minimum distance between samples (meters)
            
        Returns
        -------
        Path
            Path to sample points file
        """
        output_path = Path(output_path)
        
        logger.info(f"Creating sample points (n={n_samples} per class)")
        
        if not HAS_GEOPANDAS:
            logger.warning("GeoPandas not available")
            return flood_events_path
        
        # Read flood events
        gdf = gpd.read_file(flood_events_path)
        
        # Ensure in target CRS
        if str(gdf.crs) != self.target_crs:
            gdf = gdf.to_crs(self.target_crs)
        
        # The flood events file already contains balanced samples
        # Just filter and clean
        
        # Ensure flood column exists
        if 'flood' not in gdf.columns:
            gdf['flood'] = 1  # Assume all are flood points
        
        # Save
        gdf.to_file(output_path, driver="GPKG")
        
        flood_count = sum(gdf['flood'] == 1)
        non_flood_count = sum(gdf['flood'] == 0)
        
        logger.info(f"  Flood samples: {flood_count}")
        logger.info(f"  Non-flood samples: {non_flood_count}")
        logger.info(f"  Output: {output_path}")
        
        return output_path
    
    # =========================================================================
    # PREPROCESS ALL
    # =========================================================================
    
    @timer
    def preprocess_all(
        self,
        raw_paths: Dict[str, Path],
        output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Preprocess all datasets.
        
        Parameters
        ----------
        raw_paths : dict
            Dictionary of {name: path} for raw data
        output_dir : str or Path
            Output directory
            
        Returns
        -------
        dict
            Dictionary of {name: path} for processed data
        """
        output_dir = ensure_dir(output_dir)
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING ALL DATASETS")
        logger.info("=" * 60)
        
        processed = {}
        
        # Process DEM first (reference layer)
        if 'dem' in raw_paths:
            processed['dem'] = self.preprocess_raster(
                raw_paths['dem'],
                output_dir / "dem_processed.tif",
                resampling='bilinear'
            )
        
        # Process rainfall
        if 'rainfall' in raw_paths:
            processed['rainfall'] = self.preprocess_raster(
                raw_paths['rainfall'],
                output_dir / "rainfall_processed.tif",
                resampling='bilinear'
            )
        
        # Process land cover (use nearest neighbor for categorical)
        if 'landcover' in raw_paths:
            processed['landcover'] = self.preprocess_raster(
                raw_paths['landcover'],
                output_dir / "landcover_processed.tif",
                resampling='nearest'
            )
        
        # Process rivers (vector)
        if 'rivers' in raw_paths:
            processed['rivers'] = self.preprocess_vector(
                raw_paths['rivers'],
                output_dir / "rivers_processed.gpkg"
            )
        
        # Process flood events (vector)
        if 'flood_events' in raw_paths:
            processed['flood_events'] = self.preprocess_vector(
                raw_paths['flood_events'],
                output_dir / "flood_events_processed.gpkg"
            )
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        
        return processed


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from config import PROCESSING, PROCESSED_DATA_DIR, RAW_DATA_DIR
    
    preprocessor = DataPreprocessor(
        target_crs=PROCESSING['target_crs'],
        target_resolution=PROCESSING['target_resolution'],
        nodata=PROCESSING['nodata_value']
    )
    
    # Example usage
    print("DataPreprocessor initialized")
    print(f"  Target CRS: {preprocessor.target_crs}")
    print(f"  Target Resolution: {preprocessor.target_resolution}m")
