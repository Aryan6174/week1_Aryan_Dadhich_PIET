# src/feature_engineering.py
"""
Feature Engineering Module for Tesla ML Pipeline
Creates advanced features for modeling and forecasting
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Comprehensive feature engineering for Tesla time series data
    """
    
    def __init__(self):
        self.feature_names = []
        self.engineered_features = []
        
    def create_time_features(self, df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
        """
        Create time-based features from date column
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            
        Returns:
            DataFrame with time features added
        """
        df_feat = df.copy()
        
        print("\n🔧 Creating Time-Based Features...")
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_feat[date_col]):
            df_feat[date_col] = pd.to_datetime(df_feat[date_col])
        
        # Extract time components
        df_feat['Year'] = df_feat[date_col].dt.year
        df_feat['Quarter'] = df_feat[date_col].dt.quarter
        df_feat['Month'] = df_feat[date_col].dt.month
        df_feat['Day_of_Year'] = df_feat[date_col].dt.dayofyear
        df_feat['Week_of_Year'] = df_feat[date_col].dt.isocalendar().week
        df_feat['Is_Quarter_End'] = df_feat[date_col].dt.is_quarter_end.astype(int)
        df_feat['Is_Year_End'] = df_feat[date_col].dt.is_year_end.astype(int)
        
        # Cyclical encoding for seasonality
        df_feat['Quarter_Sin'] = np.sin(2 * np.pi * df_feat['Quarter'] / 4)
        df_feat['Quarter_Cos'] = np.cos(2 * np.pi * df_feat['Quarter'] / 4)
        df_feat['Month_Sin'] = np.sin(2 * np.pi * df_feat['Month'] / 12)
        df_feat['Month_Cos'] = np.cos(2 * np.pi * df_feat['Month'] / 12)
        
        time_features = ['Year', 'Quarter', 'Month', 'Day_of_Year', 'Week_of_Year',
                        'Is_Quarter_End', 'Is_Year_End', 'Quarter_Sin', 'Quarter_Cos',
                        'Month_Sin', 'Month_Cos']
        
        self.engineered_features.extend(time_features)
        
        print(f"   ✓ Created {len(time_features)} time-based features")
        for feat in time_features:
            print(f"      • {feat}")
        
        return df_feat
    
    def create_lag_features(self, df: pd.DataFrame, 
                           target_cols: List[str],
                           lags: List[int] = [1, 2, 3, 4]) -> pd.DataFrame:
        """
        Create lag features for time series
        
        Args:
            df: Input DataFrame
            target_cols: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        df_feat = df.copy()
        
        print(f"\n🔧 Creating Lag Features (lags: {lags})...")
        
        lag_features = []
        for col in target_cols:
            if col in df_feat.columns:
                for lag in lags:
                    feat_name = f'{col}_Lag_{lag}'
                    df_feat[feat_name] = df_feat[col].shift(lag)
                    lag_features.append(feat_name)
                    
        print(f"   ✓ Created {len(lag_features)} lag features")
        self.engineered_features.extend(lag_features)
        
        return df_feat
    
    def create_rolling_features(self, df: pd.DataFrame,
                               target_cols: List[str],
                               windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Args:
            df: Input DataFrame
            target_cols: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        df_feat = df.copy()
        
        print(f"\n🔧 Creating Rolling Window Features (windows: {windows})...")
        
        rolling_features = []
        for col in target_cols:
            if col in df_feat.columns:
                for window in windows:
                    # Rolling mean
                    feat_name = f'{col}_RollingMean_{window}'
                    df_feat[feat_name] = df_feat[col].rolling(window=window, min_periods=1).mean()
                    rolling_features.append(feat_name)
                    
                    # Rolling std
                    feat_name = f'{col}_RollingStd_{window}'
                    df_feat[feat_name] = df_feat[col].rolling(window=window, min_periods=1).std()
                    rolling_features.append(feat_name)
                    
                    # Rolling min/max
                    feat_name = f'{col}_RollingMin_{window}'
                    df_feat[feat_name] = df_feat[col].rolling(window=window, min_periods=1).min()
                    rolling_features.append(feat_name)
                    
                    feat_name = f'{col}_RollingMax_{window}'
                    df_feat[feat_name] = df_feat[col].rolling(window=window, min_periods=1).max()
                    rolling_features.append(feat_name)
        
        print(f"   ✓ Created {len(rolling_features)} rolling features")
        self.engineered_features.extend(rolling_features)
        
        return df_feat
    
    def create_difference_features(self, df: pd.DataFrame,
                                   target_cols: List[str],
                                   periods: List[int] = [1, 4]) -> pd.DataFrame:
        """
        Create difference and percentage change features
        
        Args:
            df: Input DataFrame
            target_cols: Columns to create differences for
            periods: List of periods for differencing
            
        Returns:
            DataFrame with difference features added
        """
        df_feat = df.copy()
        
        print(f"\n🔧 Creating Difference & Growth Features...")
        
        diff_features = []
        for col in target_cols:
            if col in df_feat.columns:
                for period in periods:
                    # Absolute difference
                    feat_name = f'{col}_Diff_{period}'
                    df_feat[feat_name] = df_feat[col].diff(period)
                    diff_features.append(feat_name)
                    
                    # Percentage change
                    feat_name = f'{col}_PctChange_{period}'
                    df_feat[feat_name] = df_feat[col].pct_change(period) * 100
                    diff_features.append(feat_name)
        
        print(f"   ✓ Created {len(diff_features)} difference/growth features")
        self.engineered_features.extend(diff_features)
        
        return df_feat
    
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create business-specific ratio features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with ratio features added
        """
        df_feat = df.copy()
        
        print(f"\n🔧 Creating Business Ratio Features...")
        
        ratio_features = []
        
        # Production to Delivery ratio
        if 'Production_Units' in df_feat.columns and 'Estimated_Deliveries' in df_feat.columns:
            df_feat['Production_to_Delivery_Ratio'] = (
                df_feat['Production_Units'] / (df_feat['Estimated_Deliveries'] + 1)
            )
            ratio_features.append('Production_to_Delivery_Ratio')
            
            # Inventory change
            df_feat['Inventory_Change'] = df_feat['Production_Units'] - df_feat['Estimated_Deliveries']
            ratio_features.append('Inventory_Change')
        
        # Revenue calculation
        if 'Estimated_Deliveries' in df_feat.columns and 'Avg_Price_USD' in df_feat.columns:
            df_feat['Revenue_Millions'] = (
                (df_feat['Estimated_Deliveries'] * df_feat['Avg_Price_USD']) / 1e6
            )
            ratio_features.append('Revenue_Millions')
            
            df_feat['Revenue_per_Vehicle'] = df_feat['Avg_Price_USD']
            ratio_features.append('Revenue_per_Vehicle')
        
        # Battery efficiency metrics
        if 'Range_km' in df_feat.columns and 'Battery_Capacity_kWh' in df_feat.columns:
            df_feat['Range_per_kWh'] = (
                df_feat['Range_km'] / (df_feat['Battery_Capacity_kWh'] + 1)
            )
            ratio_features.append('Range_per_kWh')
        
        # CO2 efficiency
        if 'CO2_Saved_tons' in df_feat.columns and 'Estimated_Deliveries' in df_feat.columns:
            df_feat['CO2_per_Vehicle'] = (
                df_feat['CO2_Saved_tons'] / (df_feat['Estimated_Deliveries'] + 1)
            )
            ratio_features.append('CO2_per_Vehicle')
        
        print(f"   ✓ Created {len(ratio_features)} ratio features")
        for feat in ratio_features:
            print(f"      • {feat}")
        
        self.engineered_features.extend(ratio_features)
        
        return df_feat
    def create_trend_features(self, df: pd.DataFrame,
                             target_cols: List[str]) -> pd.DataFrame:
        """
        Create trend indicator features
        
        Args:
            df: Input DataFrame
            target_cols: Columns to create trends for
            
        Returns:
            DataFrame with trend features added
        """
        df_feat = df.copy()
        
        print(f"\n🔧 Creating Trend Indicator Features...")
        
        trend_features = []
        for col in target_cols:
            if col in df_feat.columns:
                # Momentum (current vs previous)
                feat_name = f'{col}_Momentum'
                df_feat[feat_name] = (
                    (df_feat[col] > df_feat[col].shift(1)).astype(int)
                )
                trend_features.append(feat_name)
                
                # Acceleration (rate of change of change)
                feat_name = f'{col}_Acceleration'
                pct_change = df_feat[col].pct_change()
                df_feat[feat_name] = pct_change.diff()
                trend_features.append(feat_name)
        
        print(f"   ✓ Created {len(trend_features)} trend features")
        self.engineered_features.extend(trend_features)
        
        return df_feat
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features added
        """
        df_feat = df.copy()
        
        print(f"\n🔧 Creating Interaction Features...")
        
        interaction_features = []
        
        # Year × Quarter interaction
        if 'Year' in df_feat.columns and 'Quarter' in df_feat.columns:
            df_feat['Year_Quarter'] = df_feat['Year'] * 10 + df_feat['Quarter']
            interaction_features.append('Year_Quarter')
        
        # Battery × Range interaction
        if 'Battery_Capacity_kWh' in df_feat.columns and 'Range_km' in df_feat.columns:
            df_feat['Battery_Range_Product'] = (
                df_feat['Battery_Capacity_kWh'] * df_feat['Range_km']
            )
            interaction_features.append('Battery_Range_Product')
        
        print(f"   ✓ Created {len(interaction_features)} interaction features")
        self.engineered_features.extend(interaction_features)
        
        return df_feat
    def engineer_features(self, df: pd.DataFrame,
                        date_col: str = 'Date',
                        target_cols: Optional[List[str]] = None,
                        lag_periods: List[int] = [1, 2, 3, 4],
                        rolling_windows: List[int] = [3, 6, 12],
                        diff_periods: List[int] = [1, 4]) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            target_cols: Target columns for lag/rolling features
            lag_periods: Periods for lag features
            rolling_windows: Windows for rolling features
            diff_periods: Periods for difference features
            
        Returns:
            DataFrame with all engineered features
        """
        print("\n" + "="*80)
        print("STARTING FEATURE ENGINEERING PIPELINE")
        print("="*80)
        
        if target_cols is None:
            target_cols = ['Production_Units', 'Estimated_Deliveries']  # UPDATED
        
        df_feat = df.copy()
        
        # Create all feature types
        df_feat = self.create_time_features(df_feat, date_col)
        df_feat = self.create_lag_features(df_feat, target_cols, lag_periods)
        df_feat = self.create_rolling_features(df_feat, target_cols, rolling_windows)
        df_feat = self.create_difference_features(df_feat, target_cols, diff_periods)
        df_feat = self.create_ratio_features(df_feat)
        df_feat = self.create_trend_features(df_feat, target_cols)
        df_feat = self.create_interaction_features(df_feat)
        
        # Fill NaN values created by lag/rolling operations
        df_feat = df_feat.bfill().ffill()        
        print("\n" + "="*80)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*80)
        print(f"Original features: {df.shape[1]}")
        print(f"Engineered features: {len(self.engineered_features)}")
        print(f"Total features: {df_feat.shape[1]}")
        print("="*80 + "\n")
        
        return df_feat
    def get_feature_importance_groups(self) -> dict:
        """
        Group engineered features by type for analysis
        
        Returns:
            Dictionary of feature groups
        """
        groups = {
            'time_features': [f for f in self.engineered_features if any(x in f for x in ['Year', 'Quarter', 'Month', 'Week', 'Day', 'Sin', 'Cos'])],
            'lag_features': [f for f in self.engineered_features if 'Lag' in f],
            'rolling_features': [f for f in self.engineered_features if 'Rolling' in f],
            'difference_features': [f for f in self.engineered_features if 'Diff' in f or 'PctChange' in f],
            'ratio_features': [f for f in self.engineered_features if 'Ratio' in f or 'Mix' in f or 'Rate' in f or 'Change' in f or 'Revenue_per' in f],
            'trend_features': [f for f in self.engineered_features if 'Momentum' in f or 'Acceleration' in f],
            'interaction_features': [f for f in self.engineered_features if 'Year_Quarter' in f or 'Effective' in f]
        }
        return groups


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from src.data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(random_state=42)
    df = preprocessor.load_data('data/raw/tesla_data.csv')
    df_clean = preprocessor.preprocess(df, date_columns=['Date'])
    
    # Engineer features
    engineer = FeatureEngineer()
    df_featured = engineer.engineer_features(
        df_clean,
        date_col='Date',
        target_cols=['Production', 'Deliveries'],
        lag_periods=[1, 2, 3, 4],
        rolling_windows=[3, 6, 12],
        diff_periods=[1, 4]
    )
    
    # Save
    output_path = 'data/processed/tesla_featured.csv'
    df_featured.to_csv(output_path, index=False)
    print(f"✓ Featured data saved to {output_path}")
    
    # Show feature groups
    print("\n📊 Feature Groups:")
    groups = engineer.get_feature_importance_groups()
    for group_name, features in groups.items():
        print(f"\n{group_name} ({len(features)}):")
        for feat in features[:5]:  # Show first 5
            print(f"   • {feat}")
        if len(features) > 5:
            print(f"   ... and {len(features) - 5} more")