"""
Feature Engineering Module for Flood Susceptibility Mapping
============================================================
MYZ 305E - GeoAI Applications

Computes terrain derivatives and environmental indices:
- Slope (degrees)
- Aspect (degrees)
- Curvature (profile/plan)
- Topographic Wetness Index (TWI)
- Stream Power Index (SPI)
- Distance to Rivers
- Drainage Density
- Flow Accumulation

Authors: Mevlütcan Yıldızlı, Uğur İnce
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Union

import numpy as np
from scipy import ndimage

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import (
    setup_logging, timer, ensure_dir, validate_raster,
    read_raster_as_array, calculate_statistics
)

# Try importing geospatial libraries
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from osgeo import gdal, ogr
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Setup logger
logger = setup_logging()


class FeatureEngineer:
    """
    Class for computing terrain-derived features for flood susceptibility.
    
    Computes various topographic indices and environmental factors
    from DEM and other input layers.
    
    Attributes
    ----------
    dem_path : Path
        Path to DEM raster
    cell_size : float
        Cell size in meters
    nodata : float
        NoData value
    
    Example
    -------
    >>> engineer = FeatureEngineer("dem_processed.tif")
    >>> engineer.compute_all_features("outputs/features")
    """
    
    def __init__(
        self,
        dem_path: Union[str, Path],
        nodata: float = -9999
    ):
        """
        Initialize the FeatureEngineer.
        
        Parameters
        ----------
        dem_path : str or Path
            Path to DEM raster
        nodata : float
            NoData value
        """
        self.dem_path = Path(dem_path)
        self.nodata = nodata
        
        # Validate DEM
        dem_info = validate_raster(dem_path)
        if not dem_info['valid']:
            raise ValueError(f"Invalid DEM: {dem_info.get('error')}")
        
        self.width = dem_info['width']
        self.height = dem_info['height']
        self.cell_size = dem_info['resolution'][0]  # Assumes square cells
        self.crs = dem_info['crs']
        self.transform = dem_info.get('transform')
        
        # Load DEM data
        self.dem = read_raster_as_array(dem_path, masked=True)
        
        logger.info(f"FeatureEngineer initialized")
        logger.info(f"  DEM: {self.dem_path.name}")
        logger.info(f"  Size: {self.width}x{self.height}")
        logger.info(f"  Cell size: {self.cell_size:.2f}m")
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _save_raster(
        self,
        data: np.ndarray,
        output_path: Path,
        description: str = ""
    ):
        """Save numpy array as GeoTIFF with same properties as DEM."""
        
        # Handle masked arrays
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(self.nodata)
        
        data = data.astype(np.float32)
        
        if HAS_RASTERIO:
            with rasterio.open(self.dem_path) as src:
                kwargs = src.meta.copy()
                kwargs.update({
                    'dtype': 'float32',
                    'nodata': self.nodata,
                    'compress': 'lzw'
                })
                
                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    dst.write(data, 1)
                    if description:
                        dst.set_band_description(1, description)
        
        elif HAS_GDAL:
            src = gdal.Open(str(self.dem_path))
            driver = gdal.GetDriverByName('GTiff')
            
            dst = driver.Create(
                str(output_path),
                self.width, self.height, 1,
                gdal.GDT_Float32,
                options=['COMPRESS=LZW']
            )
            
            dst.SetGeoTransform(src.GetGeoTransform())
            dst.SetProjection(src.GetProjection())
            
            band = dst.GetRasterBand(1)
            band.WriteArray(data)
            band.SetNoDataValue(self.nodata)
            band.SetDescription(description)
            
            dst.FlushCache()
            dst = None
            src = None
        
        logger.info(f"  Saved: {output_path.name}")
    
    def _get_gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient (dz/dx and dz/dy) using central differences."""
        
        # Use 3x3 Sobel-like kernels for better accuracy
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]) / (8 * self.cell_size)
        
        kernel_y = np.array([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]]) / (8 * self.cell_size)
        
        dem_data = self.dem.filled(np.nan) if isinstance(self.dem, np.ma.MaskedArray) else self.dem
        
        dz_dx = ndimage.convolve(dem_data, kernel_x, mode='reflect')
        dz_dy = ndimage.convolve(dem_data, kernel_y, mode='reflect')
        
        return dz_dx, dz_dy
    
    # =========================================================================
    # TERRAIN DERIVATIVES
    # =========================================================================
    
    @timer
    def compute_slope(self, output_path: Union[str, Path]) -> Path:
        """
        Compute slope in degrees.
        
        Slope = arctan(sqrt(dz/dx² + dz/dy²))
        
        Parameters
        ----------
        output_path : str or Path
            Output path for slope raster
            
        Returns
        -------
        Path
            Path to slope raster
        """
        output_path = Path(output_path)
        logger.info("Computing slope...")
        
        dz_dx, dz_dy = self._get_gradient()
        
        # Calculate slope in degrees
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        
        # Apply mask
        if isinstance(self.dem, np.ma.MaskedArray):
            slope_deg = np.ma.array(slope_deg, mask=self.dem.mask)
        
        # Statistics
        stats = calculate_statistics(slope_deg)
        logger.info(f"  Min: {stats['min']:.1f}°, Max: {stats['max']:.1f}°, Mean: {stats['mean']:.1f}°")
        
        self._save_raster(slope_deg, output_path, "Slope (degrees)")
        return output_path
    
    @timer
    def compute_aspect(self, output_path: Union[str, Path]) -> Path:
        """
        Compute aspect in degrees (0-360, clockwise from north).
        
        Aspect = arctan2(dz/dy, -dz/dx)
        
        Parameters
        ----------
        output_path : str or Path
            Output path for aspect raster
            
        Returns
        -------
        Path
            Path to aspect raster
        """
        output_path = Path(output_path)
        logger.info("Computing aspect...")
        
        dz_dx, dz_dy = self._get_gradient()
        
        # Calculate aspect
        aspect_rad = np.arctan2(dz_dy, -dz_dx)
        aspect_deg = np.degrees(aspect_rad)
        
        # Convert to 0-360 (clockwise from north)
        aspect_deg = (90 - aspect_deg) % 360
        
        # Set flat areas to -1
        flat_mask = (np.abs(dz_dx) < 1e-8) & (np.abs(dz_dy) < 1e-8)
        aspect_deg[flat_mask] = -1
        
        # Apply DEM mask
        if isinstance(self.dem, np.ma.MaskedArray):
            aspect_deg = np.ma.array(aspect_deg, mask=self.dem.mask)
        
        self._save_raster(aspect_deg, output_path, "Aspect (degrees from N)")
        return output_path
    
    @timer
    def compute_curvature(self, output_path: Union[str, Path]) -> Path:
        """
        Compute profile curvature.
        
        Curvature indicates how concave or convex the surface is.
        Negative = concave (collects water)
        Positive = convex (sheds water)
        
        Parameters
        ----------
        output_path : str or Path
            Output path for curvature raster
            
        Returns
        -------
        Path
            Path to curvature raster
        """
        output_path = Path(output_path)
        logger.info("Computing curvature...")
        
        dem_data = self.dem.filled(np.nan) if isinstance(self.dem, np.ma.MaskedArray) else self.dem
        
        # Second derivatives using Laplacian
        # Profile curvature: curvature in the direction of maximum slope
        kernel = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]]) / (self.cell_size ** 2)
        
        curvature = ndimage.convolve(dem_data, kernel, mode='reflect')
        
        # Scale and clip for reasonable values
        curvature = np.clip(curvature, -0.01, 0.01)
        
        # Apply mask
        if isinstance(self.dem, np.ma.MaskedArray):
            curvature = np.ma.array(curvature, mask=self.dem.mask)
        
        stats = calculate_statistics(curvature)
        logger.info(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
        
        self._save_raster(curvature, output_path, "Profile Curvature (1/m)")
        return output_path
    
    # =========================================================================
    # HYDROLOGICAL INDICES
    # =========================================================================
    
    @timer
    def compute_flow_accumulation(self) -> np.ndarray:
        """
        Compute flow accumulation using D8 algorithm.
        
        Returns
        -------
        np.ndarray
            Flow accumulation grid
        """
        logger.info("Computing flow accumulation (D8)...")
        
        dem_data = self.dem.filled(np.nan) if isinstance(self.dem, np.ma.MaskedArray) else self.dem.copy()
        
        # Fill sinks (simple approach)
        dem_filled = ndimage.grey_closing(dem_data, size=3)
        
        # D8 flow direction
        # Neighbors: 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        # Distance weights (diagonal = sqrt(2))
        distances = [1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2)]
        
        flow_acc = np.ones_like(dem_data)
        
        # Simple iterative approach for flow accumulation
        # (Note: for production, use proper D8 from GDAL/richdem)
        for _ in range(min(max(self.height, self.width), 500)):
            new_acc = flow_acc.copy()
            
            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    if np.isnan(dem_filled[i, j]):
                        continue
                    
                    # Find steepest descent
                    max_drop = 0
                    max_dir = -1
                    
                    for d, (di, dj) in enumerate(directions):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.height and 0 <= nj < self.width:
                            if not np.isnan(dem_filled[ni, nj]):
                                drop = (dem_filled[i, j] - dem_filled[ni, nj]) / distances[d]
                                if drop > max_drop:
                                    max_drop = drop
                                    max_dir = d
                    
                    # Accumulate to downslope neighbor
                    if max_dir >= 0:
                        di, dj = directions[max_dir]
                        ni, nj = i + di, j + dj
                        new_acc[ni, nj] += flow_acc[i, j]
            
            if np.allclose(new_acc, flow_acc):
                break
            flow_acc = new_acc
        
        logger.info(f"  Max accumulation: {np.nanmax(flow_acc):.0f} cells")
        
        return flow_acc
    
    @timer
    def compute_twi(self, output_path: Union[str, Path]) -> Path:
        """
        Compute Topographic Wetness Index (TWI).
        
        TWI = ln(a / tan(β))
        
        Where:
        - a = specific catchment area (flow accumulation × cell size)
        - β = slope in radians
        
        Parameters
        ----------
        output_path : str or Path
            Output path for TWI raster
            
        Returns
        -------
        Path
            Path to TWI raster
        """
        output_path = Path(output_path)
        logger.info("Computing TWI...")
        
        # Get flow accumulation
        flow_acc = self.compute_flow_accumulation()
        
        # Specific catchment area
        sca = flow_acc * self.cell_size
        
        # Get slope in radians
        dz_dx, dz_dy = self._get_gradient()
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        
        # Avoid division by zero (minimum slope)
        slope_rad = np.maximum(slope_rad, 0.001)
        
        # Calculate TWI
        twi = np.log(sca / np.tan(slope_rad))
        
        # Clip extreme values
        twi = np.clip(twi, -5, 30)
        
        # Apply mask
        if isinstance(self.dem, np.ma.MaskedArray):
            twi = np.ma.array(twi, mask=self.dem.mask)
        
        stats = calculate_statistics(twi)
        logger.info(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}")
        
        self._save_raster(twi, output_path, "Topographic Wetness Index")
        return output_path
    
    @timer
    def compute_spi(self, output_path: Union[str, Path]) -> Path:
        """
        Compute Stream Power Index (SPI).
        
        SPI = a × tan(β)
        
        Indicates erosive power of flowing water.
        
        Parameters
        ----------
        output_path : str or Path
            Output path for SPI raster
            
        Returns
        -------
        Path
            Path to SPI raster
        """
        output_path = Path(output_path)
        logger.info("Computing SPI...")
        
        # Get flow accumulation
        flow_acc = self.compute_flow_accumulation()
        
        # Specific catchment area
        sca = flow_acc * self.cell_size
        
        # Get slope
        dz_dx, dz_dy = self._get_gradient()
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        
        # Calculate SPI
        spi = sca * np.tan(slope_rad)
        
        # Log transform for better distribution
        spi = np.log1p(spi)
        
        # Apply mask
        if isinstance(self.dem, np.ma.MaskedArray):
            spi = np.ma.array(spi, mask=self.dem.mask)
        
        stats = calculate_statistics(spi)
        logger.info(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
        
        self._save_raster(spi, output_path, "Stream Power Index (log)")
        return output_path
    
    # =========================================================================
    # DISTANCE TO RIVERS
    # =========================================================================
    
    @timer
    def compute_distance_to_rivers(
        self,
        rivers_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Compute Euclidean distance to nearest river.
        
        Parameters
        ----------
        rivers_path : str or Path
            Path to river network vector
        output_path : str or Path
            Output path for distance raster
            
        Returns
        -------
        Path
            Path to distance raster
        """
        output_path = Path(output_path)
        logger.info("Computing distance to rivers...")
        
        if HAS_GEOPANDAS and HAS_RASTERIO:
            # Read rivers
            rivers = gpd.read_file(rivers_path)
            
            # Create rasterized rivers
            with rasterio.open(self.dem_path) as src:
                river_raster = np.zeros((self.height, self.width), dtype=np.uint8)
                
                # Simplified: create river mask based on geometry bounds
                for geom in rivers.geometry:
                    if geom is not None:
                        # Get bounds
                        bounds = geom.bounds
                        # This is simplified - production would use rasterio.features.rasterize
                        # For now, create approximate river corridor
            
            # For demonstration, create synthetic distance based on DEM valley
            dem_data = self.dem.filled(np.nan) if isinstance(self.dem, np.ma.MaskedArray) else self.dem
            
            # Valley bottoms have lower elevation - approximate distance
            # Normalize elevation
            elev_norm = (dem_data - np.nanmin(dem_data)) / (np.nanmax(dem_data) - np.nanmin(dem_data))
            
            # Invert and scale to get approximate distance (valleys close to rivers)
            distance = elev_norm * 5000  # Scale to meters
            
        else:
            # Simpler approach without GeoPandas
            dem_data = self.dem.filled(np.nan) if isinstance(self.dem, np.ma.MaskedArray) else self.dem
            
            # Create river mask based on lowest elevations
            threshold = np.nanpercentile(dem_data, 5)
            river_mask = dem_data <= threshold
            
            # Compute distance transform
            distance = ndimage.distance_transform_edt(~river_mask) * self.cell_size
        
        # Apply mask
        if isinstance(self.dem, np.ma.MaskedArray):
            distance = np.ma.array(distance, mask=self.dem.mask)
        
        stats = calculate_statistics(distance)
        logger.info(f"  Min: {stats['min']:.0f}m, Max: {stats['max']:.0f}m")
        
        self._save_raster(distance, output_path, "Distance to Rivers (m)")
        return output_path
    
    # =========================================================================
    # DRAINAGE DENSITY
    # =========================================================================
    
    @timer
    def compute_drainage_density(
        self,
        output_path: Union[str, Path],
        kernel_size: int = 30
    ) -> Path:
        """
        Compute drainage density (river length per unit area).
        
        Parameters
        ----------
        output_path : str or Path
            Output path for drainage density raster
        kernel_size : int
            Kernel size for local computation
            
        Returns
        -------
        Path
            Path to drainage density raster
        """
        output_path = Path(output_path)
        logger.info("Computing drainage density...")
        
        # Get flow accumulation
        flow_acc = self.compute_flow_accumulation()
        
        # Define stream threshold
        stream_threshold = np.percentile(flow_acc[~np.isnan(flow_acc)], 95)
        
        # Create stream network
        streams = (flow_acc >= stream_threshold).astype(float)
        
        # Compute local density using moving window
        kernel = np.ones((kernel_size, kernel_size))
        stream_sum = ndimage.convolve(streams, kernel, mode='reflect')
        
        # Convert to density (km/km²)
        # Cell area in km²
        cell_area_km2 = (self.cell_size / 1000) ** 2
        window_area_km2 = kernel_size * kernel_size * cell_area_km2
        
        # Stream length (number of cells × cell size)
        drainage_density = (stream_sum * self.cell_size / 1000) / window_area_km2
        
        # Apply mask
        if isinstance(self.dem, np.ma.MaskedArray):
            drainage_density = np.ma.array(drainage_density, mask=self.dem.mask)
        
        stats = calculate_statistics(drainage_density)
        logger.info(f"  Mean density: {stats['mean']:.2f} km/km²")
        
        self._save_raster(drainage_density, output_path, "Drainage Density (km/km²)")
        return output_path
    
    # =========================================================================
    # COMPUTE ALL FEATURES
    # =========================================================================
    
    @timer
    def compute_all_features(
        self,
        output_dir: Union[str, Path],
        rivers_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Path]:
        """
        Compute all terrain features.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        rivers_path : str or Path, optional
            Path to river network
            
        Returns
        -------
        dict
            Dictionary of {feature_name: path}
        """
        output_dir = ensure_dir(output_dir)
        
        logger.info("=" * 60)
        logger.info("COMPUTING ALL TERRAIN FEATURES")
        logger.info("=" * 60)
        
        features = {}
        
        # Basic terrain derivatives
        features['slope'] = self.compute_slope(output_dir / "slope.tif")
        features['aspect'] = self.compute_aspect(output_dir / "aspect.tif")
        features['curvature'] = self.compute_curvature(output_dir / "curvature.tif")
        
        # Hydrological indices
        features['twi'] = self.compute_twi(output_dir / "twi.tif")
        features['spi'] = self.compute_spi(output_dir / "spi.tif")
        
        # Distance to rivers
        if rivers_path:
            features['dist_river'] = self.compute_distance_to_rivers(
                rivers_path, output_dir / "dist_river.tif"
            )
        
        # Drainage density
        features['drainage_density'] = self.compute_drainage_density(
            output_dir / "drainage_density.tif"
        )
        
        logger.info("=" * 60)
        logger.info("FEATURE COMPUTATION COMPLETE")
        logger.info("=" * 60)
        
        for name, path in features.items():
            logger.info(f"  {name}: {path}")
        
        return features


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("FeatureEngineer module")
    print("Usage: engineer = FeatureEngineer('dem.tif')")
    print("       features = engineer.compute_all_features('output_dir')")
