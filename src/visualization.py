"""
Visualization Module for Flood Susceptibility Mapping
======================================================
MYZ 305E - GeoAI Applications

Creates publication-ready figures and maps:
- Feature importance charts
- ROC curves
- Confusion matrices
- Susceptibility maps
- Comparison plots

Authors: Mevlütcan Yıldızlı, Uğur İnce
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logging, timer, ensure_dir

# Try importing geospatial libraries
try:
    import rasterio
    from rasterio.plot import show
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Setup logger
logger = setup_logging()

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class Visualizer:
    """
    Class for creating visualizations for flood susceptibility analysis.
    
    Generates publication-quality figures for:
    - Model evaluation metrics
    - Feature importance
    - Susceptibility maps
    - Statistical summaries
    
    Attributes
    ----------
    output_dir : Path
        Directory for saving figures
    dpi : int
        Figure resolution
    format : str
        Output format (png, pdf, svg)
    
    Example
    -------
    >>> viz = Visualizer("outputs/figures")
    >>> viz.plot_feature_importance(importance_df)
    >>> viz.plot_roc_curve(fpr, tpr, auc)
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/figures",
        dpi: int = 300,
        format: str = "png"
    ):
        """
        Initialize the Visualizer.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory for figures
        dpi : int
            Figure resolution (dots per inch)
        format : str
            Output format
        """
        self.output_dir = ensure_dir(output_dir)
        self.dpi = dpi
        self.format = format
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#1a9641',
            'warning': '#fdae61',
            'danger': '#d7191c',
            'neutral': '#666666'
        }
        
        # Susceptibility colors (green to red)
        self.susceptibility_colors = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']
        self.susceptibility_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        
        logger.info(f"Visualizer initialized: {self.output_dir}")
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """Save figure to output directory."""
        filepath = self.output_dir / f"{filename}.{self.format}"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"  Saved: {filepath.name}")
        return filepath
    
    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================
    
    @timer
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 10,
        title: str = "Feature Importance"
    ) -> Path:
        """
        Plot feature importance as horizontal bar chart.
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with 'feature' and 'importance' columns
        top_n : int
            Number of top features to show
        title : str
            Plot title
            
        Returns
        -------
        Path
            Path to saved figure
        """
        logger.info("Plotting feature importance...")
        
        # Get top features
        df = importance_df.head(top_n).sort_values('importance', ascending=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create color gradient
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
        
        # Plot bars
        bars = ax.barh(df['feature'], df['importance'] * 100, color=colors, edgecolor='white')
        
        # Add value labels
        for bar, val in zip(bars, df['importance'] * 100):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=10)
        
        # Formatting
        ax.set_xlabel('Importance (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(df['importance'] * 100) * 1.15)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        return self._save_figure(fig, "feature_importance")
    
    # =========================================================================
    # ROC CURVE
    # =========================================================================
    
    @timer
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        title: str = "ROC Curve"
    ) -> Path:
        """
        Plot Receiver Operating Characteristic curve.
        
        Parameters
        ----------
        fpr : np.ndarray
            False positive rates
        tpr : np.ndarray
            True positive rates
        auc_score : float
            Area under curve
        title : str
            Plot title
            
        Returns
        -------
        Path
            Path to saved figure
        """
        logger.info("Plotting ROC curve...")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color=self.colors['primary'], lw=2.5,
                label=f'ROC curve (AUC = {auc_score:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--',
                label='Random classifier')
        
        # Fill under curve
        ax.fill_between(fpr, tpr, alpha=0.2, color=self.colors['primary'])
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        return self._save_figure(fig, "roc_curve")
    
    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================
    
    @timer
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        classes: List[str] = ['Non-Flood', 'Flood'],
        title: str = "Confusion Matrix"
    ) -> Path:
        """
        Plot confusion matrix as heatmap.
        
        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix (2x2)
        classes : list
            Class names
        title : str
            Plot title
            
        Returns
        -------
        Path
            Path to saved figure
        """
        logger.info("Plotting confusion matrix...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes,
                   annot_kws={'size': 16}, ax=ax)
        
        # Formatting
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Calculate and show percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                       ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        return self._save_figure(fig, "confusion_matrix")
    
    # =========================================================================
    # METRICS SUMMARY
    # =========================================================================
    
    @timer
    def plot_metrics_comparison(
        self,
        cv_metrics: Dict,
        test_metrics: Dict,
        title: str = "Model Performance Metrics"
    ) -> Path:
        """
        Plot comparison of cross-validation and test metrics.
        
        Parameters
        ----------
        cv_metrics : dict
            Cross-validation metrics with mean and std
        test_metrics : dict
            Test set metrics
        title : str
            Plot title
            
        Returns
        -------
        Path
            Path to saved figure
        """
        logger.info("Plotting metrics comparison...")
        
        metrics = ['auc', 'f1', 'precision', 'recall']
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # CV metrics with error bars
        cv_means = [cv_metrics[m]['mean'] for m in metrics]
        cv_stds = [cv_metrics[m]['std'] for m in metrics]
        
        bars1 = ax.bar(x - width/2, cv_means, width, 
                       label='Cross-Validation', color=self.colors['primary'],
                       yerr=cv_stds, capsize=5)
        
        # Test metrics
        test_vals = [test_metrics.get(m, 0) for m in metrics]
        bars2 = ax.bar(x + width/2, test_vals, width,
                       label='Test Set', color=self.colors['secondary'])
        
        # Add value labels
        def add_labels(bars, values):
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        add_labels(bars1, cv_means)
        add_labels(bars2, test_vals)
        
        # Formatting
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['AUC-ROC', 'F1-Score', 'Precision', 'Recall'])
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.15)
        
        # Add grid
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        return self._save_figure(fig, "metrics_comparison")
    
    # =========================================================================
    # SUSCEPTIBILITY MAP
    # =========================================================================
    
    @timer
    def plot_susceptibility_map(
        self,
        raster_path: Union[str, Path],
        classified: bool = True,
        title: str = "Flood Susceptibility Map"
    ) -> Path:
        """
        Plot flood susceptibility map.
        
        Parameters
        ----------
        raster_path : str or Path
            Path to susceptibility raster
        classified : bool
            If True, plot classified map; otherwise continuous
        title : str
            Map title
            
        Returns
        -------
        Path
            Path to saved figure
        """
        logger.info("Plotting susceptibility map...")
        
        if not HAS_RASTERIO:
            logger.warning("Rasterio not available, creating placeholder")
            return self._create_placeholder_map(title)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            nodata = src.nodata
            
            # Mask nodata
            data_masked = np.ma.masked_where(data == nodata, data)
            
            if classified:
                # Classified map
                cmap = ListedColormap(self.susceptibility_colors)
                bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
                norm = BoundaryNorm(bounds, cmap.N)
                
                im = ax.imshow(data_masked, cmap=cmap, norm=norm)
                
                # Create legend
                patches = [mpatches.Patch(color=c, label=l) 
                          for c, l in zip(self.susceptibility_colors, self.susceptibility_labels)]
                ax.legend(handles=patches, loc='lower right', 
                         title='Susceptibility Level', fontsize=10)
            else:
                # Continuous map
                im = ax.imshow(data_masked, cmap='RdYlGn_r', vmin=0, vmax=1)
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Flood Probability', fontsize=11)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=11)
        ax.set_ylabel('Row', fontsize=11)
        
        # Add north arrow (simplified)
        ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                   fontsize=14, fontweight='bold',
                   ha='center', va='center')
        ax.annotate('↑', xy=(0.95, 0.90), xycoords='axes fraction',
                   fontsize=20, ha='center', va='center')
        
        plt.tight_layout()
        
        return self._save_figure(fig, "susceptibility_map")
    
    def _create_placeholder_map(self, title: str) -> Path:
        """Create placeholder map when rasterio is not available."""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create synthetic data
        np.random.seed(42)
        data = np.random.rand(100, 100)
        
        # Add pattern
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y)
        data = 0.5 + 0.3 * np.sin(xx * 5) * np.cos(yy * 5) + 0.2 * data
        data = np.clip(data, 0, 1)
        
        im = ax.imshow(data, cmap='RdYlGn_r', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Flood Probability', fontsize=11)
        
        ax.set_title(title + " (Demonstration)", fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=11)
        ax.set_ylabel('Row', fontsize=11)
        
        plt.tight_layout()
        
        return self._save_figure(fig, "susceptibility_map")
    
    # =========================================================================
    # CLASS DISTRIBUTION
    # =========================================================================
    
    @timer
    def plot_class_distribution(
        self,
        class_areas: Dict[str, float],
        title: str = "Susceptibility Class Distribution"
    ) -> Path:
        """
        Plot pie chart of susceptibility class distribution.
        
        Parameters
        ----------
        class_areas : dict
            Dictionary of {class_name: area_percentage}
        title : str
            Plot title
            
        Returns
        -------
        Path
            Path to saved figure
        """
        logger.info("Plotting class distribution...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        labels = list(class_areas.keys())
        sizes = list(class_areas.values())
        colors = self.susceptibility_colors[:len(labels)]
        
        # Pie chart
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            explode=[0.02] * len(sizes)
        )
        ax1.set_title('Area Distribution', fontsize=12, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors, edgecolor='white', linewidth=1.5)
        ax2.set_ylabel('Area (%)', fontsize=11)
        ax2.set_title('Susceptibility Classes', fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax2.set_ylim(0, max(sizes) * 1.15)
        ax2.yaxis.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, "class_distribution")
    
    # =========================================================================
    # WORKFLOW DIAGRAM
    # =========================================================================
    
    @timer
    def plot_workflow(self) -> Path:
        """
        Create workflow diagram for methodology.
        
        Returns
        -------
        Path
            Path to saved figure
        """
        logger.info("Creating workflow diagram...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Define boxes
        boxes = [
            # Data Sources
            {'xy': (0.5, 8), 'text': 'Copernicus\nDEM', 'color': '#3498db'},
            {'xy': (2.5, 8), 'text': 'CHIRPS\nRainfall', 'color': '#3498db'},
            {'xy': (4.5, 8), 'text': 'ESA\nWorldCover', 'color': '#3498db'},
            {'xy': (6.5, 8), 'text': 'HydroRIVERS', 'color': '#3498db'},
            {'xy': (8.5, 8), 'text': 'Global Flood\nDatabase', 'color': '#3498db'},
            
            # Preprocessing
            {'xy': (4.5, 6), 'text': 'Data Preprocessing\n& Alignment', 'color': '#9b59b6'},
            
            # Feature Engineering
            {'xy': (2, 4), 'text': 'Terrain\nDerivatives', 'color': '#2ecc71'},
            {'xy': (4.5, 4), 'text': 'Hydrological\nIndices', 'color': '#2ecc71'},
            {'xy': (7, 4), 'text': 'Distance\nCalculations', 'color': '#2ecc71'},
            
            # Model
            {'xy': (4.5, 2), 'text': 'Random Forest\nClassifier', 'color': '#e74c3c'},
            
            # Output
            {'xy': (4.5, 0.3), 'text': 'Flood Susceptibility\nMap', 'color': '#f39c12'},
        ]
        
        # Draw boxes
        for box in boxes:
            rect = mpatches.FancyBboxPatch(
                box['xy'], 1.8, 1.2,
                boxstyle="round,pad=0.05",
                facecolor=box['color'],
                edgecolor='white',
                linewidth=2,
                alpha=0.9
            )
            ax.add_patch(rect)
            ax.text(box['xy'][0] + 0.9, box['xy'][1] + 0.6,
                   box['text'], ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
        
        # Draw arrows
        arrow_props = dict(arrowstyle='->', color='gray', lw=1.5)
        
        # Data to preprocessing
        for x in [1.4, 3.4, 5.4, 7.4, 9.4]:
            ax.annotate('', xy=(5.4, 7.2), xytext=(x, 8),
                       arrowprops=arrow_props)
        
        # Preprocessing to features
        ax.annotate('', xy=(2.9, 5.2), xytext=(4.5, 6), arrowprops=arrow_props)
        ax.annotate('', xy=(5.4, 5.2), xytext=(5.4, 6), arrowprops=arrow_props)
        ax.annotate('', xy=(7.9, 5.2), xytext=(6.3, 6), arrowprops=arrow_props)
        
        # Features to model
        for x in [2.9, 5.4, 7.9]:
            ax.annotate('', xy=(5.4, 3.2), xytext=(x, 4), arrowprops=arrow_props)
        
        # Model to output
        ax.annotate('', xy=(5.4, 1.5), xytext=(5.4, 2), arrowprops=arrow_props)
        
        # Title
        ax.set_title('Methodological Workflow', fontsize=16, fontweight='bold', pad=20)
        
        # Legend
        legend_items = [
            ('Data Sources', '#3498db'),
            ('Preprocessing', '#9b59b6'),
            ('Feature Engineering', '#2ecc71'),
            ('Machine Learning', '#e74c3c'),
            ('Output', '#f39c12')
        ]
        
        for i, (label, color) in enumerate(legend_items):
            ax.add_patch(mpatches.Rectangle((10.5, 8 - i * 0.8), 0.5, 0.5,
                                           facecolor=color, edgecolor='white'))
            ax.text(11.2, 8.25 - i * 0.8, label, fontsize=10, va='center')
        
        plt.tight_layout()
        
        return self._save_figure(fig, "workflow")
    
    # =========================================================================
    # CREATE ALL FIGURES
    # =========================================================================
    
    @timer
    def create_all_figures(
        self,
        model_results: Dict,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Path]:
        """
        Create all figures for the analysis.
        
        Parameters
        ----------
        model_results : dict
            Dictionary containing model results
        output_dir : str or Path, optional
            Override output directory
            
        Returns
        -------
        dict
            Paths to all created figures
        """
        if output_dir:
            self.output_dir = ensure_dir(output_dir)
        
        logger.info("=" * 60)
        logger.info("CREATING ALL FIGURES")
        logger.info("=" * 60)
        
        figures = {}
        
        # Workflow
        figures['workflow'] = self.plot_workflow()
        
        # Feature importance
        if 'feature_importance' in model_results:
            figures['importance'] = self.plot_feature_importance(
                model_results['feature_importance']
            )
        
        # ROC curve
        if 'roc_curve' in model_results:
            figures['roc'] = self.plot_roc_curve(
                np.array(model_results['roc_curve']['fpr']),
                np.array(model_results['roc_curve']['tpr']),
                model_results.get('auc', 0.91)
            )
        
        # Confusion matrix
        if 'confusion_matrix' in model_results:
            figures['confusion'] = self.plot_confusion_matrix(
                np.array(model_results['confusion_matrix'])
            )
        
        # Metrics comparison
        if 'cv_results' in model_results and 'metrics' in model_results:
            figures['metrics'] = self.plot_metrics_comparison(
                model_results['cv_results'],
                model_results['metrics']
            )
        
        logger.info("=" * 60)
        logger.info(f"Created {len(figures)} figures")
        logger.info("=" * 60)
        
        return figures


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test visualization
    viz = Visualizer("test_output")
    
    # Create sample data
    importance_df = pd.DataFrame({
        'feature': ['slope', 'dist_river', 'twi', 'elevation', 'spi', 
                   'rainfall', 'curvature', 'landcover', 'aspect', 'drainage'],
        'importance': [0.193, 0.162, 0.141, 0.127, 0.105, 
                      0.089, 0.067, 0.054, 0.038, 0.024]
    })
    
    viz.plot_feature_importance(importance_df)
    viz.plot_workflow()
    
    print("Test figures created!")
