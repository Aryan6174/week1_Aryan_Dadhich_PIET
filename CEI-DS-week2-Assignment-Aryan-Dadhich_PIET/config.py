# config.py
"""
Configuration file for Tesla ML Pipeline
"""

import os
from pathlib import Path

# Project Structure
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models' / 'saved_models'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature Engineering
LAG_PERIODS = [1, 2, 3, 4]
ROLLING_WINDOWS = [3, 6, 12]

# Forecasting
FORECAST_HORIZON = 12

# Model Registry
MODELS = {
    'linear_regression': 'LinearRegression',
    'ridge': 'Ridge',
    'lasso': 'Lasso',
    'random_forest': 'RandomForestRegressor',
    'xgboost': 'XGBoostRegressor'
}

print("Configuration loaded successfully!")