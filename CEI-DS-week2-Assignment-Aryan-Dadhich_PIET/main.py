# main.py
"""
Main execution script for Tesla ML Pipeline
Run complete end-to-end pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import modules
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.forecast import TimeSeriesForecaster
from src.evaluate import ModelEvaluator
from config import *


def main():
    """
    Execute complete ML pipeline
    """
    print("\n" + "="*80)
    print("TESLA ML PIPELINE - COMPLETE EXECUTION")
    print("="*80)
    
    # Step 1: Data Preprocessing
    print("\n[1/5] DATA PREPROCESSING...")
    preprocessor = DataPreprocessor(random_state=RANDOM_SEED)
    # Ensure the file exists in data/raw/tesla_data.csv or update path
    df_raw = preprocessor.load_data(str(RAW_DATA_DIR / 'tesla_data.csv'))
    df_clean = preprocessor.preprocess(df_raw, create_date_column=True) 
    df_clean.to_csv(PROCESSED_DATA_DIR / 'tesla_clean.csv', index=False)
    
    # Step 2: Feature Engineering
    print("\n[2/5] FEATURE ENGINEERING...")
    engineer = FeatureEngineer()
    df_featured = engineer.engineer_features(
        df_clean,
        date_col='Date',
        target_cols=['Production_Units', 'Estimated_Deliveries'], 
        lag_periods=LAG_PERIODS,
        rolling_windows=ROLLING_WINDOWS
    )
    df_featured.to_csv(PROCESSED_DATA_DIR / 'tesla_featured.csv', index=False)
    
    # Step 3: Model Training
    print("\n[3/5] MODEL TRAINING...")
    trainer = ModelTrainer(random_state=RANDOM_SEED)

    TARGET = 'Estimated_Deliveries' 
    EXCLUDE_FEATURES = ['Date', 'Estimated_Deliveries', 'Year', 'Month',
                    'Region', 'Model'] 

    feature_cols = [col for col in df_featured.columns if col not in EXCLUDE_FEATURES]
    
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(
        df_featured,
        target_col=TARGET,
        feature_cols=feature_cols,
        test_size=TEST_SIZE
    )
    
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    comparison_df = trainer.create_comparison_table()
    comparison_df.to_csv(REPORTS_DIR / 'model_comparison.csv', index=False)
    
    # Step 4: Time Series Forecasting
    print("\n[4/5] TIME SERIES FORECASTING...")
    forecaster = TimeSeriesForecaster(random_state=RANDOM_SEED)
    
    df_ts = df_clean.copy()
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    df_ts = df_ts.sort_values('Date')
    
    # --- UPDATED COLUMN NAMES BELOW ---
    
    # ARIMA
    # Changed 'Deliveries' to 'Estimated_Deliveries'
    ts_deliveries = forecaster.prepare_time_series(df_ts, 'Date', 'Estimated_Deliveries', 'MS')
    train_ts, test_ts = forecaster.train_test_split_ts(ts_deliveries, test_size=12) # Use 12 for monthly
    arima_result = forecaster.forecast_arima(train_ts, test_ts, 
                                             order=(1, 1, 1),
                                             forecast_horizon=FORECAST_HORIZON)
    
    # Prophet
    # Changed 'Deliveries' to 'Estimated_Deliveries'
    prophet_result = forecaster.forecast_prophet(df_ts, 'Date', 'Estimated_Deliveries',
                                                 forecast_horizon=FORECAST_HORIZON)
    
    # XGBoost TS
    # Changed 'Deliveries' to 'Estimated_Deliveries'
    xgb_ts_result = forecaster.forecast_xgboost_ts(df_ts, 'Date', 'Estimated_Deliveries',
                                                   forecast_horizon=FORECAST_HORIZON)
    
    forecast_comparison = forecaster.compare_forecasts()
    forecast_comparison.to_csv(REPORTS_DIR / 'forecast_comparison.csv', index=False)
    
    # Step 5: Save Models
    print("\n[5/5] SAVING MODELS...")
    best_model_name = comparison_df.iloc[0]['Model']
    trainer.save_model(results[best_model_name]['model'],
                      f'best_model_{best_model_name.lower().replace(" ", "_")}',
                      str(MODELS_DIR))
    
    import joblib
    joblib.dump(trainer.scaler, MODELS_DIR / 'scaler.joblib')
    joblib.dump(feature_names, MODELS_DIR / 'feature_names.joblib')
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"\n✓ Best Regression Model: {best_model_name}")
    print(f"   Test RMSE: {results[best_model_name]['test_metrics']['RMSE']:,.2f}")
    print(f"   Test R²: {results[best_model_name]['test_metrics']['R2']:.4f}")
    
    best_forecast = forecast_comparison.iloc[0]['Model']
    print(f"\n✓ Best Forecast Model: {best_forecast}")
    print(f"   Test RMSE: {forecast_comparison.iloc[0]['Test_RMSE']:,.2f}")
    
    print(f"\n✓ All outputs saved to:")
    print(f"   • Data: {PROCESSED_DATA_DIR}")
    print(f"   • Models: {MODELS_DIR}")
    print(f"   • Reports: {REPORTS_DIR}")
    print(f"   • Figures: {FIGURES_DIR}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()