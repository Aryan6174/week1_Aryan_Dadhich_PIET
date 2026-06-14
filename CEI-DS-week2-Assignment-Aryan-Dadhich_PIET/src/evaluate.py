# src/evaluate.py
"""
Model Evaluation and Visualization Module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization
    """
    
    def __init__(self, save_dir: str = 'reports/figures'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_predictions(self, y_true, y_pred, title: str = 'Predictions vs Actual',
                        save_name: str = None):
        """
        Plot predictions vs actual values
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=100)
        axes[0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values', fontsize=12)
        axes[0].set_ylabel('Predicted Values', fontsize=12)
        axes[0].set_title(f'{title} - Scatter Plot', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=100)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title(f'{title} - Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                               title: str = 'Feature Importance',
                               top_n: int = 20,
                               save_name: str = None):
        """
        Plot feature importance
        """
        plt.figure(figsize=(12, 8))
        
        top_features = feature_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                             metric: str = 'Test_RMSE',
                             save_name: str = None):
        """
        Plot model comparison
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        comparison_df = comparison_df.sort_values(metric)
        
        colors = sns.color_palette("husl", len(comparison_df))
        bars = ax.barh(comparison_df['Model'], comparison_df[metric], color=colors)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:,.2f}',
                   ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_forecast(self, historical: pd.Series,
                     forecast: np.array,
                     forecast_dates: pd.DatetimeIndex,
                     test: pd.Series = None,
                     confidence_intervals: pd.DataFrame = None,
                     title: str = 'Forecast',
                     save_name: str = None):
        """
        Plot time series forecast
        """
        plt.figure(figsize=(15, 6))
        
        # Plot historical data
        plt.plot(historical.index, historical.values, 
                label='Historical', linewidth=2, marker='o')
        
        # Plot test data if available
        if test is not None:
            plt.plot(test.index, test.values,
                    label='Actual (Test)', linewidth=2, marker='o', color='green')
        
        # Plot forecast
        plt.plot(forecast_dates, forecast,
                label='Forecast', linewidth=2, marker='s', 
                linestyle='--', color='red')
        
        # Plot confidence intervals if available
        if confidence_intervals is not None:
            plt.fill_between(confidence_intervals.index,
                           confidence_intervals['mean_ci_lower'],
                           confidence_intervals['mean_ci_upper'],
                           alpha=0.3, color='red', label='95% Confidence Interval')
        
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Value', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    print("Model Evaluator module loaded successfully!")