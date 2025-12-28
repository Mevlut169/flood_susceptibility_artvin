# Methodology Documentation
## Flood Susceptibility Mapping with Random Forest

### MYZ 305E - GeoAI Applications
### Istanbul Technical University, Fall 2025

---

## 1. Study Area

**Location:** Artvin Province, Türkiye  
**Coordinates:** 40.7°N - 41.7°N, 41.0°E - 42.5°E  
**Area:** 7,436 km²  
**CRS:** UTM Zone 37N (EPSG:32637)

### 1.1 Rationale for Study Area Selection

Artvin Province was selected due to:
- High flood susceptibility from steep terrain and heavy rainfall
- Historical flood events (2010, 2012, 2015, 2021)
- Active Çoruh River basin
- Diverse topography for robust model testing

---

## 2. Data Sources

### 2.1 Copernicus DEM GLO-30
- **Resolution:** 30 meters
- **Source:** European Space Agency (ESA) / Copernicus Programme
- **URL:** https://portal.opentopography.org/
- **Variables Derived:** Elevation, Slope, Aspect, Curvature, TWI, SPI

### 2.2 CHIRPS v2.0
- **Resolution:** ~5.5 km (0.05°)
- **Source:** Climate Hazards Center, UC Santa Barbara
- **URL:** https://data.chc.ucsb.edu/products/CHIRPS-2.0/
- **Variable:** Annual precipitation (mm/year)
- **Period:** 2000-2023 mean

### 2.3 ESA WorldCover 2021
- **Resolution:** 10 meters
- **Source:** European Space Agency
- **URL:** https://esa-worldcover.org/
- **Variable:** Land cover classification

### 2.4 HydroRIVERS
- **Format:** Vector (shapefile/GeoPackage)
- **Source:** WWF HydroSHEDS
- **URL:** https://www.hydrosheds.org/products/hydrorivers
- **Variable:** River network for distance calculation

### 2.5 Global Flood Database
- **Source:** Cloud to Street
- **URL:** https://global-flood-database.cloudtostreet.ai/
- **Variable:** Historical flood extent polygons
- **Period:** 2000-2018

---

## 3. Conditioning Factors

### 3.1 Terrain-Derived Factors

| Factor | Formula | Flood Relation |
|--------|---------|----------------|
| Elevation | Direct from DEM | Lower areas accumulate water |
| Slope | arctan(√(∂z/∂x² + ∂z/∂y²)) | Low slope = water ponding |
| Aspect | arctan2(∂z/∂y, -∂z/∂x) | Affects moisture distribution |
| Curvature | ∂²z/∂x² + ∂²z/∂y² | Concave areas collect water |

### 3.2 Hydrological Indices

**Topographic Wetness Index (TWI):**
```
TWI = ln(a / tan(β))
```
Where:
- a = specific catchment area (flow accumulation × cell size)
- β = slope in radians

**Stream Power Index (SPI):**
```
SPI = a × tan(β)
```

### 3.3 Proximity Factors

**Distance to Rivers:**
- Euclidean distance from each cell to nearest river segment
- Calculated using scipy.ndimage.distance_transform_edt()

**Drainage Density:**
- River length per unit area (km/km²)
- Computed using moving window analysis

### 3.4 Environmental Factors

**Land Cover:**
- Categorical variable (10 classes)
- One-hot encoded for model input

**Annual Rainfall:**
- Mean annual precipitation from CHIRPS
- Resampled to 30m resolution

---

## 4. Preprocessing Pipeline

### 4.1 Coordinate Reference System
- All layers reprojected to **EPSG:32637** (UTM Zone 37N)
- Ensures consistent distance calculations in meters

### 4.2 Resolution Harmonization
- Target resolution: **30 meters**
- Resampling methods:
  - Continuous data: Bilinear interpolation
  - Categorical data: Nearest neighbor

### 4.3 Extent Alignment
- All rasters clipped to common extent
- NoData value: -9999

---

## 5. Sample Generation

### 5.1 Positive Samples (Flood)
- Source: Global Flood Database historical events
- Count: 847 points
- Method: Random sampling within flood extent polygons
- Minimum separation: 500 meters (reduce spatial autocorrelation)

### 5.2 Negative Samples (Non-Flood)
- Count: 847 points (balanced dataset)
- Method: Random sampling outside flood extents
- Constraints:
  - Minimum 500m from flood extents
  - Stratified by land cover type

---

## 6. Random Forest Model

### 6.1 Algorithm Selection Rationale

Random Forest was selected due to:
1. Non-parametric nature (no distribution assumptions)
2. Handles non-linear relationships
3. Robust to outliers and noise
4. Built-in feature importance
5. Proven performance in similar studies

### 6.2 Hyperparameters

```python
RandomForestClassifier(
    n_estimators=300,      # Number of trees
    max_depth=20,          # Maximum tree depth
    max_features='sqrt',   # Features per split
    min_samples_split=5,   # Min samples to split
    min_samples_leaf=3,    # Min samples in leaf
    class_weight='balanced', # Handle imbalance
    random_state=42        # Reproducibility
)
```

### 6.3 Training Protocol

1. **Data Split:** 70% training, 30% testing (stratified)
2. **Feature Scaling:** StandardScaler normalization
3. **Cross-Validation:** 5-fold stratified CV
4. **Evaluation Metrics:** AUC-ROC, F1-Score, Precision, Recall

---

## 7. Susceptibility Classification

### 7.1 Probability Output
- Range: 0.0 - 1.0
- Represents flood occurrence probability

### 7.2 Classification Scheme

| Class | Name | Percentile Range |
|-------|------|------------------|
| 1 | Very Low | 0 - 20th |
| 2 | Low | 20th - 40th |
| 3 | Moderate | 40th - 60th |
| 4 | High | 60th - 80th |
| 5 | Very High | 80th - 100th |

### 7.3 Classification Method
- Natural breaks (Jenks) for optimal class separation
- Validated with equal interval and quantile methods

---

## 8. Validation

### 8.1 Internal Validation
- 5-fold stratified cross-validation
- Metrics: AUC-ROC, F1, Precision, Recall

### 8.2 External Validation
- 30% holdout test set
- Confusion matrix analysis
- ROC curve analysis

### 8.3 Expected Performance

| Metric | CV (mean ± std) | Test Set |
|--------|-----------------|----------|
| AUC-ROC | 0.90 ± 0.02 | 0.91 |
| F1-Score | 0.86 ± 0.03 | 0.87 |
| Precision | 0.84 ± 0.02 | 0.85 |
| Recall | 0.88 ± 0.03 | 0.89 |

---

## 9. Software Requirements

### 9.1 Core Dependencies
- Python ≥ 3.9
- NumPy ≥ 1.24
- Pandas ≥ 2.0
- Scikit-learn ≥ 1.3

### 9.2 Geospatial Libraries
- GDAL ≥ 3.6
- Rasterio ≥ 1.3
- GeoPandas ≥ 0.13
- Shapely ≥ 2.0

### 9.3 Visualization
- Matplotlib ≥ 3.7
- Seaborn ≥ 0.12

---

## 10. References

1. Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.

2. Chen, W., et al. (2019). Flood susceptibility modelling using novel hybrid approach of reduced-error pruning trees with bagging and random subspace ensembles. Journal of Hydrology, 575, 864-873.

3. Tellman, B., et al. (2021). Satellite imaging reveals increased proportion of population exposed to floods. Nature, 596(7870), 80-86.

4. Yuksek, O., et al. (2013). Assessment of flood hazards and flood risk in the Eastern Black Sea basin, Turkey. Natural Hazards, 66, 571-586.

5. Tehrany, M.S., et al. (2015). Flood susceptibility mapping using a novel ensemble weights-of-evidence and support vector machine models in GIS. Journal of Hydrology, 512, 332-343.

---

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Authors:** Mevlütcan Yıldızlı, Uğur İnce
