"""
Machine Learning Model Module for Flood Susceptibility Mapping
===============================================================
MYZ 305E - GeoAI Applications

Implements Random Forest classifier for flood susceptibility:
- Feature extraction from rasters
- Model training with cross-validation
- Hyperparameter tuning
- Model evaluation and metrics
- Susceptibility prediction
- Feature importance analysis

Authors: Mevlütcan Yıldızlı, Uğur İnce
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logging, timer, ensure_dir, read_raster_as_array

# Try importing geospatial libraries
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Setup logger
logger = setup_logging()


class FloodSusceptibilityModel:
    """
    Random Forest model for flood susceptibility mapping.
    
    Trains a Random Forest classifier on environmental features
    to predict flood susceptibility across the study area.
    
    Attributes
    ----------
    model : RandomForestClassifier
        Trained Random Forest model
    scaler : StandardScaler
        Feature scaler
    feature_names : list
        Names of input features
    metrics : dict
        Evaluation metrics
    
    Example
    -------
    >>> model = FloodSusceptibilityModel()
    >>> model.prepare_training_data(feature_paths, samples_path)
    >>> model.train()
    >>> model.evaluate()
    >>> model.predict_susceptibility(output_path)
    """
    
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 20,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the FloodSusceptibilityModel.
        
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of trees
        random_state : int
            Random seed for reproducibility
        **kwargs
            Additional parameters for RandomForestClassifier
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Model parameters
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_features': kwargs.get('max_features', 'sqrt'),
            'min_samples_split': kwargs.get('min_samples_split', 5),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 3),
            'class_weight': kwargs.get('class_weight', 'balanced'),
            'random_state': random_state,
            'n_jobs': kwargs.get('n_jobs', -1),
            'verbose': kwargs.get('verbose', 1)
        }
        
        # Initialize model
        self.model = RandomForestClassifier(**self.model_params)
        self.scaler = StandardScaler()
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        self.feature_paths = {}
        
        # Results
        self.metrics = {}
        self.feature_importance = None
        self.cv_results = None
        
        logger.info("FloodSusceptibilityModel initialized")
        logger.info(f"  Trees: {n_estimators}, Max Depth: {max_depth}")
    
    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    
    @timer
    def prepare_training_data(
        self,
        feature_paths: Dict[str, Path],
        samples_path: Union[str, Path],
        test_size: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from feature rasters and sample points.
        
        Parameters
        ----------
        feature_paths : dict
            Dictionary of {feature_name: raster_path}
        samples_path : str or Path
            Path to sample points (with 'flood' column)
        test_size : float
            Proportion for test set
            
        Returns
        -------
        tuple
            (X, y) arrays
        """
        logger.info("Preparing training data...")
        
        self.feature_paths = feature_paths
        self.feature_names = list(feature_paths.keys())
        
        logger.info(f"  Features: {self.feature_names}")
        
        # Load sample points
        if HAS_GEOPANDAS:
            samples = gpd.read_file(samples_path)
        else:
            # Fallback: load as CSV
            samples = pd.read_csv(samples_path)
        
        n_samples = len(samples)
        logger.info(f"  Sample points: {n_samples}")
        
        # Extract feature values at sample points
        X = np.zeros((n_samples, len(self.feature_names)))
        
        for i, (name, path) in enumerate(feature_paths.items()):
            logger.info(f"  Extracting: {name}")
            
            values = self._extract_values_at_points(path, samples)
            X[:, i] = values
        
        # Get labels
        y = samples['flood'].values
        
        # Handle missing values
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"  Valid samples after filtering: {len(y)}")
        logger.info(f"  Flood: {sum(y == 1)}, Non-flood: {sum(y == 0)}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"  Training samples: {len(self.y_train)}")
        logger.info(f"  Test samples: {len(self.y_test)}")
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return X, y
    
    def _extract_values_at_points(
        self,
        raster_path: Union[str, Path],
        points: Union[gpd.GeoDataFrame, pd.DataFrame]
    ) -> np.ndarray:
        """Extract raster values at point locations."""
        
        if HAS_RASTERIO and HAS_GEOPANDAS and hasattr(points, 'geometry'):
            with rasterio.open(raster_path) as src:
                values = []
                for geom in points.geometry:
                    try:
                        row, col = src.index(geom.x, geom.y)
                        if 0 <= row < src.height and 0 <= col < src.width:
                            val = src.read(1)[row, col]
                        else:
                            val = np.nan
                    except:
                        val = np.nan
                    values.append(val)
                return np.array(values)
        else:
            # Fallback: create synthetic values for demonstration
            n_points = len(points)
            np.random.seed(42)
            return np.random.randn(n_points)
    
    # =========================================================================
    # MODEL TRAINING
    # =========================================================================
    
    @timer
    def train(self, cv_folds: int = 5) -> Dict:
        """
        Train the Random Forest model with cross-validation.
        
        Parameters
        ----------
        cv_folds : int
            Number of cross-validation folds
            
        Returns
        -------
        dict
            Cross-validation results
        """
        if self.X_train is None:
            raise ValueError("No training data. Call prepare_training_data first.")
        
        logger.info("=" * 60)
        logger.info("TRAINING RANDOM FOREST MODEL")
        logger.info("=" * 60)
        
        # Cross-validation
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'auc': cross_val_score(self.model, self.X_train, self.y_train, 
                                   cv=cv, scoring='roc_auc'),
            'f1': cross_val_score(self.model, self.X_train, self.y_train, 
                                  cv=cv, scoring='f1'),
            'precision': cross_val_score(self.model, self.X_train, self.y_train, 
                                         cv=cv, scoring='precision'),
            'recall': cross_val_score(self.model, self.X_train, self.y_train, 
                                      cv=cv, scoring='recall')
        }
        
        self.cv_results = {
            metric: {'mean': scores.mean(), 'std': scores.std(), 'scores': scores.tolist()}
            for metric, scores in cv_scores.items()
        }
        
        logger.info("\nCross-validation Results:")
        logger.info("-" * 40)
        for metric, results in self.cv_results.items():
            logger.info(f"  {metric.upper()}: {results['mean']:.3f} ± {results['std']:.3f}")
        
        # Train final model on all training data
        logger.info("\nTraining final model...")
        self.model.fit(self.X_train, self.y_train)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nFeature Importance:")
        logger.info("-" * 40)
        for _, row in self.feature_importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']*100:.1f}%")
        
        return self.cv_results
    
    @timer
    def tune_hyperparameters(
        self,
        param_grid: Optional[Dict] = None,
        cv_folds: int = 3
    ) -> Dict:
        """
        Tune hyperparameters using grid search.
        
        Parameters
        ----------
        param_grid : dict, optional
            Parameter grid for search
        cv_folds : int
            Number of CV folds for tuning
            
        Returns
        -------
        dict
            Best parameters and score
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, 25],
                'min_samples_leaf': [1, 2, 3, 5]
            }
        
        logger.info("Tuning hyperparameters...")
        
        grid_search = GridSearchCV(
            RandomForestClassifier(
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            param_grid,
            cv=cv_folds,
            scoring='roc_auc',
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        logger.info(f"\nBest Parameters: {grid_search.best_params_}")
        logger.info(f"Best AUC: {grid_search.best_score_:.3f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    # =========================================================================
    # MODEL EVALUATION
    # =========================================================================
    
    @timer
    def evaluate(self) -> Dict:
        """
        Evaluate model on test set.
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        if self.X_test is None:
            raise ValueError("No test data available.")
        
        logger.info("=" * 60)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 60)
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'auc': roc_auc_score(self.y_test, y_prob)
        }
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        self.metrics['classification_report'] = report
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(self.y_test, y_prob)
        self.metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        # Print results
        logger.info("\nTest Set Metrics:")
        logger.info("-" * 40)
        logger.info(f"  AUC-ROC:   {self.metrics['auc']:.3f}")
        logger.info(f"  F1-Score:  {self.metrics['f1']:.3f}")
        logger.info(f"  Precision: {self.metrics['precision']:.3f}")
        logger.info(f"  Recall:    {self.metrics['recall']:.3f}")
        logger.info(f"  Accuracy:  {self.metrics['accuracy']:.3f}")
        
        logger.info("\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return self.metrics
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    @timer
    def predict_susceptibility(
        self,
        output_path: Union[str, Path],
        reference_raster: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Generate flood susceptibility map for the entire study area.
        
        Parameters
        ----------
        output_path : str or Path
            Output path for susceptibility raster
        reference_raster : str or Path, optional
            Reference raster for output properties
            
        Returns
        -------
        Path
            Path to susceptibility map
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating flood susceptibility map...")
        
        if reference_raster is None:
            reference_raster = list(self.feature_paths.values())[0]
        
        # Read reference raster for dimensions
        if HAS_RASTERIO:
            with rasterio.open(reference_raster) as ref:
                height, width = ref.height, ref.width
                profile = ref.profile.copy()
                
                # Read all features
                logger.info("  Loading feature rasters...")
                feature_stack = np.zeros((len(self.feature_names), height, width))
                
                for i, (name, path) in enumerate(self.feature_paths.items()):
                    with rasterio.open(path) as src:
                        data = src.read(1)
                        # Resample if needed
                        if data.shape != (height, width):
                            from scipy.ndimage import zoom
                            zoom_factor = (height / data.shape[0], width / data.shape[1])
                            data = zoom(data, zoom_factor, order=1)
                        feature_stack[i] = data
                
                # Reshape for prediction
                # (n_features, height, width) -> (height*width, n_features)
                n_pixels = height * width
                X_predict = feature_stack.reshape(len(self.feature_names), n_pixels).T
                
                # Handle NoData
                valid_mask = ~np.any(np.isnan(X_predict), axis=1)
                
                logger.info(f"  Valid pixels: {sum(valid_mask):,} / {n_pixels:,}")
                
                # Scale features
                X_valid = self.scaler.transform(X_predict[valid_mask])
                
                # Predict probabilities
                logger.info("  Running prediction...")
                probabilities = np.full(n_pixels, -9999, dtype=np.float32)
                probabilities[valid_mask] = self.model.predict_proba(X_valid)[:, 1]
                
                # Reshape to image
                susceptibility = probabilities.reshape(height, width)
                
                # Save raster
                profile.update(
                    dtype='float32',
                    nodata=-9999,
                    compress='lzw'
                )
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(susceptibility, 1)
                    dst.set_band_description(1, "Flood Susceptibility (probability)")
        
        else:
            # Simplified output for demonstration
            logger.warning("Rasterio not available, creating demonstration output")
            np.random.seed(42)
            susceptibility = np.random.rand(500, 500).astype(np.float32)
            np.save(output_path.with_suffix('.npy'), susceptibility)
            output_path = output_path.with_suffix('.npy')
        
        logger.info(f"  Susceptibility map saved: {output_path}")
        
        return output_path
    
    @timer
    def classify_susceptibility(
        self,
        probability_raster: Union[str, Path],
        output_path: Union[str, Path],
        n_classes: int = 5,
        method: str = 'natural_breaks'
    ) -> Path:
        """
        Classify continuous probabilities into susceptibility classes.
        
        Parameters
        ----------
        probability_raster : str or Path
            Input probability raster
        output_path : str or Path
            Output classified raster
        n_classes : int
            Number of classes
        method : str
            Classification method: 'natural_breaks', 'equal_interval', 'quantile'
            
        Returns
        -------
        Path
            Path to classified raster
        """
        output_path = Path(output_path)
        
        logger.info(f"Classifying susceptibility into {n_classes} classes...")
        
        if HAS_RASTERIO:
            with rasterio.open(probability_raster) as src:
                probs = src.read(1)
                profile = src.profile.copy()
                nodata = src.nodata
            
            # Get valid values
            valid_mask = probs != nodata
            valid_probs = probs[valid_mask]
            
            # Calculate breaks
            if method == 'equal_interval':
                breaks = np.linspace(0, 1, n_classes + 1)
            elif method == 'quantile':
                percentiles = np.linspace(0, 100, n_classes + 1)
                breaks = np.percentile(valid_probs, percentiles)
            else:  # natural_breaks (simplified Jenks)
                breaks = np.percentile(valid_probs, np.linspace(0, 100, n_classes + 1))
            
            logger.info(f"  Class breaks: {breaks}")
            
            # Classify
            classified = np.full_like(probs, 0, dtype=np.uint8)
            for i in range(n_classes):
                mask = (probs >= breaks[i]) & (probs < breaks[i + 1]) & valid_mask
                classified[mask] = i + 1
            
            # Handle edge case for max value
            classified[(probs >= breaks[-1]) & valid_mask] = n_classes
            
            # Set nodata
            classified[~valid_mask] = 0
            
            # Calculate class distribution
            for i in range(1, n_classes + 1):
                count = np.sum(classified == i)
                pct = count / np.sum(valid_mask) * 100
                logger.info(f"  Class {i}: {pct:.1f}%")
            
            # Save
            profile.update(dtype='uint8', nodata=0)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(classified, 1)
        
        else:
            logger.warning("Rasterio not available")
        
        logger.info(f"  Classified map saved: {output_path}")
        
        return output_path
    
    # =========================================================================
    # SAVE / LOAD MODEL
    # =========================================================================
    
    def save_model(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Save trained model and associated files.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
            
        Returns
        -------
        dict
            Paths to saved files
        """
        output_dir = ensure_dir(output_dir)
        
        paths = {}
        
        # Save model
        model_path = output_dir / "random_forest_model.joblib"
        joblib.dump(self.model, model_path)
        paths['model'] = model_path
        logger.info(f"  Model saved: {model_path}")
        
        # Save scaler
        scaler_path = output_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        paths['scaler'] = scaler_path
        
        # Save feature names
        features_path = output_dir / "feature_names.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        paths['features'] = features_path
        
        # Save metrics
        if self.metrics:
            metrics_path = output_dir / "metrics.json"
            # Convert numpy arrays to lists for JSON
            metrics_json = {}
            for k, v in self.metrics.items():
                if isinstance(v, np.ndarray):
                    metrics_json[k] = v.tolist()
                else:
                    metrics_json[k] = v
            with open(metrics_path, 'w') as f:
                json.dump(metrics_json, f, indent=2)
            paths['metrics'] = metrics_path
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = output_dir / "feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
            paths['importance'] = importance_path
        
        # Save CV results
        if self.cv_results:
            cv_path = output_dir / "cv_results.json"
            with open(cv_path, 'w') as f:
                json.dump(self.cv_results, f, indent=2)
            paths['cv_results'] = cv_path
        
        return paths
    
    @classmethod
    def load_model(cls, model_dir: Union[str, Path]) -> 'FloodSusceptibilityModel':
        """
        Load a saved model.
        
        Parameters
        ----------
        model_dir : str or Path
            Directory containing saved model files
            
        Returns
        -------
        FloodSusceptibilityModel
            Loaded model instance
        """
        model_dir = Path(model_dir)
        
        # Create new instance
        instance = cls()
        
        # Load model
        instance.model = joblib.load(model_dir / "random_forest_model.joblib")
        
        # Load scaler
        instance.scaler = joblib.load(model_dir / "scaler.joblib")
        
        # Load feature names
        with open(model_dir / "feature_names.json") as f:
            instance.feature_names = json.load(f)
        
        # Load metrics if exists
        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                instance.metrics = json.load(f)
        
        logger.info(f"Model loaded from: {model_dir}")
        
        return instance


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("FloodSusceptibilityModel module")
    print("\nUsage:")
    print("  model = FloodSusceptibilityModel(n_estimators=300, max_depth=20)")
    print("  model.prepare_training_data(feature_paths, samples_path)")
    print("  model.train(cv_folds=5)")
    print("  model.evaluate()")
    print("  model.predict_susceptibility('susceptibility.tif')")
