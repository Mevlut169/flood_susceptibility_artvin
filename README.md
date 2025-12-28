# 🌊 Flood Susceptibility Mapping with Random Forest
## Artvin Province, Türkiye

**MYZ 305E - GeoAI Applications**  
Istanbul Technical University, Fall 2025

**Authors:** Mevlütcan Yıldızlı, Uğur İnce

---

## 🚀 Quick Start

### 1. Install
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib
```

### 2. Run
```bash
python main.py
```

That's it! The program will:
1. Generate terrain data (DEM, Slope, TWI)
2. Show terrain maps
3. Create training samples
4. Train Random Forest model
5. Display model evaluation
6. Generate flood susceptibility map
7. Show 3D visualization
8. Save all outputs

---

## 📊 Outputs

### Figures (outputs/figures/)
- `01_terrain_analysis.png` - DEM, Slope, TWI, Distance maps
- `02_sample_locations.png` - Training sample locations
- `03_feature_distributions.png` - Feature histograms
- `04_model_evaluation.png` - Confusion matrix, ROC curve, Feature importance
- `05_flood_susceptibility_map.png` - Final susceptibility map
- `06_3d_susceptibility.png` - 3D terrain with susceptibility overlay

### Model (outputs/models/)
- `random_forest_model.joblib` - Trained model
- `scaler.joblib` - Feature scaler
- `metrics.csv` - Performance metrics

### Maps (outputs/maps/)
- `flood_susceptibility.npz` - All raster data

---

## 📈 Expected Results

| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.91 |
| F1-Score | ~0.87 |
| High Risk Area | ~23% |

---

## 🎓 Course Info

**MYZ 305E - GeoAI Applications**  
Department of Geomatics Engineering  
Istanbul Technical University  
Fall 2025
