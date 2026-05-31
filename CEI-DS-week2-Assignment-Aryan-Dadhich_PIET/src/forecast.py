# src/forecast.py
"""
Time Series Forecasting Module for Tesla ML Pipeline
Implements ARIMA, Prophet, and XGBoost time series approaches
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesForecaster:
    """
    Comprehensive time series forecasting class
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.forecasts = {}
        
    def prepare_time_series(self, df: pd.DataFrame,
                           date_col: str,
                           target_col: str = 'Estimated_Deliveries',
                           freq: str = 'M') -> pd.Series:
        """
        Prepare time series data
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            target_col: Target variable column name
            freq: Frequency ('M' for monthly, 'Q' for quarterly)
            
        Returns:
            Time series with datetime index
        """
        print("\n🔧 Preparing time series data...")
        
        if freq == 'M':
            freq = 'ME'

        print(f"DEBUG - freq received: {freq}")

        df_ts = df.copy()

        df_ts[date_col] = pd.to_datetime(df_ts[date_col])

        # Aggregate all rows for the same month
        df_ts = (
            df_ts.groupby(date_col)[target_col]
                .sum()
                .reset_index()
        )

        df_ts = df_ts.sort_values(date_col)
        df_ts = df_ts.set_index(date_col)

        ts = df_ts[target_col]
        ts = ts.asfreq('MS')
        # Fill missing periods
        ts = ts.interpolate(method='linear')
        ts = ts.bfill().ffill()
        
        print(f"Missing values after resampling: {ts.isna().sum()}")
        print(f"   ✓ Time series prepared")
        print(f"      • Length: {len(ts)}")
        print(f"      • Frequency: {freq}")
        print(f"      • Start: {ts.index[0]}")
        print(f"      • End: {ts.index[-1]}")
        
        return ts
    
    def train_test_split_ts(self, ts: pd.Series,
                           test_size: int = 12) -> tuple:
        """
        Split time series into train and test
        
        Args:
            ts: Time series
            test_size: Number of periods for test set
            
        Returns:
            Tuple of (train, test)
        """
        train = ts[:-test_size]
        test = ts[-test_size:]
        
        print(f"\n📊 Time series split:")
        print(f"   • Train: {len(train)} periods ({train.index[0]} to {train.index[-1]})")
        print(f"   • Test: {len(test)} periods ({test.index[0]} to {test.index[-1]})")
        
        return train, test
    
    def forecast_arima(self, train: pd.Series, test: pd.Series,
                      order: tuple = (1, 1, 1),
                      forecast_horizon: int = 12) -> dict:
        """
        ARIMA forecasting
        
        Args:
            train: Training time series
            test: Test time series
            order: ARIMA order (p, d, q)
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        print(f"\n📈 Training ARIMA{order}...")
        
        try:
            # Fit model
            model = ARIMA(train, order=order)
            fitted_model = model.fit()
            
            # Forecast on test set
            test_forecast = fitted_model.forecast(steps=len(test))
            
            # Forecast future
            future_forecast = fitted_model.forecast(steps=forecast_horizon)
            
            # Calculate metrics
            mae = mean_absolute_error(test, test_forecast)
            rmse = np.sqrt(mean_squared_error(test, test_forecast))
            mape = np.mean(np.abs((test - test_forecast) / (test + 1e-10))) * 100
            
            print(f"   ✓ ARIMA trained")
            print(f"      • Test MAE: {mae:,.2f}")
            print(f"      • Test RMSE: {rmse:,.2f}")
            print(f"      • Test MAPE: {mape:.2f}%")
            
            # Get confidence intervals
            forecast_result = fitted_model.get_forecast(steps=forecast_horizon)
            forecast_df = forecast_result.summary_frame()
            
            result = {
                'model': fitted_model,
                'test_forecast': test_forecast,
                'future_forecast': future_forecast,
                'forecast_df': forecast_df,
                'metrics': {'MAE': mae, 'RMSE': rmse, 'MAPE': mape},
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
            self.models['ARIMA'] = fitted_model
            self.forecasts['ARIMA'] = result
            
            return result
            
        except Exception as e:
            print(f"\n❌ ARIMA Error: {str(e)}")
            print("   Trying simpler ARIMA(1,0,0)...")
            
            # Fallback to simpler model
            model = ARIMA(train, order=(1, 0, 0))
            fitted_model = model.fit()
            
            test_forecast = fitted_model.forecast(steps=len(test))
            future_forecast = fitted_model.forecast(steps=forecast_horizon)
            
            mae = mean_absolute_error(test, test_forecast)
            rmse = np.sqrt(mean_squared_error(test, test_forecast))
            mape = np.mean(np.abs((test - test_forecast) / (test + 1e-10))) * 100
            
            forecast_result = fitted_model.get_forecast(steps=forecast_horizon)
            forecast_df = forecast_result.summary_frame()
            
            result = {
                'model': fitted_model,
                'test_forecast': test_forecast,
                'future_forecast': future_forecast,
                'forecast_df': forecast_df,
                'metrics': {'MAE': mae, 'RMSE': rmse, 'MAPE': mape},
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
            self.models['ARIMA'] = fitted_model
            self.forecasts['ARIMA'] = result
            
            print(f"   ✓ Fallback ARIMA(1,0,0) trained")
            print(f"      • Test RMSE: {rmse:,.2f}")
            
            return result
    
    def forecast_prophet(self, df: pd.DataFrame,
                        date_col: str,
                        target_col: str = 'Estimated_Deliveries',
                        forecast_horizon: int = 12,
                        test_size: int = 12) -> dict:
        """
        Prophet forecasting
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            target_col: Target column name
            forecast_horizon: Number of periods to forecast
            test_size: Number of periods for test set
            
        Returns:
            Dictionary with forecast results
        """
        print(f"\n📈 Training Prophet...")
        
        # Prepare data for Prophet
        df_prophet = df[[date_col, target_col]].copy()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet = df_prophet.sort_values('ds')
        
        # Split train/test
        train = df_prophet[:-test_size]
        test = df_prophet[-test_size:]
        
        # Fit model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        # Suppress Prophet output
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        
        model.fit(train)
        
        # Forecast on test period
        test_forecast = model.predict(test[['ds']])
        
        # Future forecast
        future = model.make_future_dataframe(periods=forecast_horizon, freq='ME')
        forecast = model.predict(future)
        
        # Calculate metrics on test set
        mae = mean_absolute_error(test['y'], test_forecast['yhat'])
        rmse = np.sqrt(mean_squared_error(test['y'], test_forecast['yhat']))
        mape = np.mean(np.abs((test['y'] - test_forecast['yhat']) / (test['y'] + 1e-10))) * 100
        
        print(f"   ✓ Prophet trained")
        print(f"      • Test MAE: {mae:,.2f}")
        print(f"      • Test RMSE: {rmse:,.2f}")
        print(f"      • Test MAPE: {mape:.2f}%")
        
        result = {
            'model': model,
            'test_forecast': test_forecast,
            'future_forecast': forecast,
            'metrics': {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        }
        
        self.models['Prophet'] = model
        self.forecasts['Prophet'] = result
        
        return result
    
    def create_time_series_features(self, df: pd.DataFrame,
                                   date_col: str) -> pd.DataFrame:
        """
        Create features for XGBoost time series approach
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            
        Returns:
            DataFrame with time series features
        """
        df_feat = df.copy()
        df_feat[date_col] = pd.to_datetime(df_feat[date_col])
        df_feat = df_feat.sort_values(date_col)
        
        # Time-based features
        df_feat['year'] = df_feat[date_col].dt.year
        df_feat['month'] = df_feat[date_col].dt.month
        df_feat['quarter'] = df_feat[date_col].dt.quarter
        df_feat['dayofyear'] = df_feat[date_col].dt.dayofyear
        
        # Trend
        df_feat['trend'] = np.arange(len(df_feat))
        
        return df_feat
    
    def forecast_xgboost_ts(self, df: pd.DataFrame,
                           date_col: str,
                           target_col: str = 'Estimated_Deliveries',
                           lag_features: list = [1, 2, 3, 6, 12],
                           forecast_horizon: int = 12,
                           test_size: int = 12) -> dict:
        """
        XGBoost time series forecasting
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            target_col: Target column name
            lag_features: List of lag periods
            forecast_horizon: Number of periods to forecast
            test_size: Number of periods for test set
            
        Returns:
            Dictionary with forecast results
        """
        print(f"\n📈 Training XGBoost Time Series Model...")
        
        # Create features
        df_feat = self.create_time_series_features(df, date_col)
        
        # Create lag features
        for lag in lag_features:
            df_feat[f'lag_{lag}'] = df_feat[target_col].shift(lag)
        
        # Create rolling features
        df_feat['rolling_mean_3'] = df_feat[target_col].rolling(window=3, min_periods=1).mean()
        df_feat['rolling_std_3'] = df_feat[target_col].rolling(window=3, min_periods=1).std()
        df_feat['rolling_mean_6'] = df_feat[target_col].rolling(window=6, min_periods=1).mean()
        
        # Drop NaN values
        df_feat = df_feat.dropna()
        
        # Features for modeling
        feature_cols = ['year', 'month', 'quarter', 'trend'] + \
                      [f'lag_{lag}' for lag in lag_features] + \
                      ['rolling_mean_3', 'rolling_std_3', 'rolling_mean_6']
        
        # Split train/test
        train = df_feat[:-test_size]
        test = df_feat[-test_size:]
        
        X_train = train[feature_cols]
        y_train = train[target_col]
        X_test = test[feature_cols]
        y_test = test[target_col]
        
        # Train model
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state,
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        # Predict on test set
        test_forecast = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, test_forecast)
        rmse = np.sqrt(mean_squared_error(y_test, test_forecast))
        mape = np.mean(np.abs((y_test - test_forecast) / (y_test + 1e-10))) * 100
        
        print(f"   ✓ XGBoost trained")
        print(f"      • Test MAE: {mae:,.2f}")
        print(f"      • Test RMSE: {rmse:,.2f}")
        print(f"      • Test MAPE: {mape:.2f}%")
        
        # Future forecast (recursive)
        future_forecasts = []
        last_known = df_feat.iloc[-1].copy()
        
        for i in range(forecast_horizon):
            # Create feature vector for next period
            next_features = {
                'year': last_known['year'] if i < 12 else last_known['year'] + 1,
                'month': (last_known['month'] + i - 1) % 12 + 1,
                'quarter': ((last_known['month'] + i - 2) % 12) // 3 + 1,
                'trend': last_known['trend'] + i + 1
            }
            
            # Use recent forecasts for lags
            for j, lag in enumerate(lag_features):
                if i >= lag:
                    # Use forecasted values
                    next_features[f'lag_{lag}'] = future_forecasts[i - lag]
                else:
                    # Use historical values
                    if lag - i - 1 < len(y_test):
                        next_features[f'lag_{lag}'] = y_test.iloc[-(lag - i)]
                    else:
                        next_features[f'lag_{lag}'] = last_known[f'lag_{lag}']
            
            # Rolling features (simplified)
            if len(future_forecasts) >= 3:
                next_features['rolling_mean_3'] = np.mean(future_forecasts[-3:])
                next_features['rolling_std_3'] = np.std(future_forecasts[-3:])
            else:
                next_features['rolling_mean_3'] = last_known['rolling_mean_3']
                next_features['rolling_std_3'] = last_known['rolling_std_3']
            
            if len(future_forecasts) >= 6:
                next_features['rolling_mean_6'] = np.mean(future_forecasts[-6:])
            else:
                next_features['rolling_mean_6'] = last_known['rolling_mean_6']
            
            # Predict
            X_next = pd.DataFrame([next_features])[feature_cols]
            next_pred = model.predict(X_next)[0]
            future_forecasts.append(next_pred)
        
        result = {
            'model': model,
            'test_forecast': test_forecast,
            'future_forecast': np.array(future_forecasts),
            'metrics': {'MAE': mae, 'RMSE': rmse, 'MAPE': mape},
            'feature_importance': pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        self.models['XGBoost_TS'] = model
        self.forecasts['XGBoost_TS'] = result
        
        return result
    
    def compare_forecasts(self) -> pd.DataFrame:
        """
        Compare all forecasting models
        
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, result in self.forecasts.items():
            comparison_data.append({
                'Model': model_name,
                'Test_MAE': result['metrics']['MAE'],
                'Test_RMSE': result['metrics']['RMSE'],
                'Test_MAPE': result['metrics']['MAPE']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.round(2)
        df_comparison = df_comparison.sort_values('Test_RMSE')
        
        return df_comparison


if __name__ == "__main__":
    print("Time Series Forecaster module loaded successfully!")