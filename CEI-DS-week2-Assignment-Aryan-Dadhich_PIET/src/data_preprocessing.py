# src/data_preprocessing.py
"""
Data Preprocessing Module for Tesla ML Pipeline
Handles data loading, cleaning, and validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for Tesla dataset
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.original_shape = None
        self.cleaned_shape = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame containing the raw data
        """
        try:
            df = pd.read_csv(filepath)
            self.original_shape = df.shape
            print(f"✓ Data loaded successfully!")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            return df
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """
        Create synthetic Tesla data matching actual dataset schema
        """
        print("Creating synthetic Tesla dataset...")
        
        np.random.seed(self.random_state)
        
        # Generate data for 2015-2025 (monthly)
        years = []
        months = []
        for year in range(2015, 2026):
            for month in range(1, 13):
                years.append(year)
                months.append(month)
        
        n_records = len(years)
        
        # Generate realistic data
        regions = np.random.choice(['North America', 'Europe', 'China', 'Other'], n_records, p=[0.4, 0.25, 0.25, 0.1])
        models = np.random.choice(['Model S', 'Model X', 'Model 3', 'Model Y'], n_records, p=[0.15, 0.15, 0.35, 0.35])
        
        # Time-based trend
        time_idx = np.arange(n_records)
        base_deliveries = 1000 + time_idx * 200 + np.random.normal(0, 500, n_records)
        base_production = base_deliveries * 1.05 + np.random.normal(0, 300, n_records)
        
        # Create DataFrame matching actual schema
        df = pd.DataFrame({
            'Year': years,
            'Month': months,
            'Region': regions,
            'Model': models,
            'Estimated_Deliveries': np.maximum(base_deliveries, 100).astype(int),
            'Production_Units': np.maximum(base_production, 100).astype(int),
            'Avg_Price_USD': np.random.uniform(40000, 90000, n_records).round(2),
            'Battery_Capacity_kWh': np.random.uniform(60, 100, n_records).round(1),
            'Range_km': np.random.uniform(350, 650, n_records).round(0),
            'CO2_Saved_tons': np.random.uniform(1, 10, n_records).round(2)
        })
        
        # Add some missing values (2% of price data)
        missing_indices = np.random.choice(n_records, size=int(n_records * 0.02), replace=False)
        df.loc[missing_indices, 'Avg_Price_USD'] = np.nan
        
        # Add some outliers
        outlier_indices = np.random.choice(n_records, size=3, replace=False)
        df.loc[outlier_indices, 'Estimated_Deliveries'] = df.loc[outlier_indices, 'Estimated_Deliveries'] * 1.5
        
        self.original_shape = df.shape
        print(f"✓ Synthetic data created: {df.shape}")
        return df
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive data information
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # Identify feature types
        info['numerical_features'] = df.select_dtypes(include=[np.number]).columns.tolist()
        info['categorical_features'] = df.select_dtypes(include=['object']).columns.tolist()
        info['datetime_features'] = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return info
    
    def print_data_summary(self, df: pd.DataFrame):
        """
        Print comprehensive data summary
        """
        info = self.get_data_info(df)
        
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print(f"\n📊 Shape: {info['shape'][0]:,} rows × {info['shape'][1]} columns")
        print(f"💾 Memory Usage: {info['memory_usage']:.2f} MB")
        print(f"🔄 Duplicates: {info['duplicates']}")
        
        print(f"\n📈 Numerical Features ({len(info['numerical_features'])}):")
        for feat in info['numerical_features']:
            print(f"   • {feat}")
        
        print(f"\n🏷️  Categorical Features ({len(info['categorical_features'])}):")
        for feat in info['categorical_features']:
            print(f"   • {feat}")
            
        print(f"\n📅 Datetime Features ({len(info['datetime_features'])}):")
        for feat in info['datetime_features']:
            print(f"   • {feat}")
        
        print("\n❓ Missing Values:")
        missing = [(k, v, info['missing_percentage'][k]) 
                   for k, v in info['missing_values'].items() if v > 0]
        if missing:
            for col, count, pct in missing:
                print(f"   • {col}: {count} ({pct:.2f}%)")
        else:
            print("   • No missing values!")
        
        print("\n" + "="*80)
        
        # Statistical Summary
        print("\n📊 STATISTICAL SUMMARY")
        print("="*80)
        print(df.describe())
        print("\n")
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        print("\n🔧 Handling Missing Values...")
        
        # Numerical columns: fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"   • {col}: Filled {df[col].isnull().sum()} missing values with median ({median_val:.2f})")
        
        # Categorical columns: fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"   • {col}: Filled {df[col].isnull().sum()} missing values with mode ({mode_val})")
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame without duplicates
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        removed_rows = initial_rows - len(df_clean)
        
        print(f"\n🔧 Removing Duplicates...")
        print(f"   • Removed {removed_rows} duplicate rows")
        
        return df_clean
    
    def detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Detect outliers using IQR method
        
        Args:
            df: Input DataFrame
            column: Column name to check
            
        Returns:
            Boolean Series indicating outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        """
        Handle outliers in numerical columns
        
        Args:
            df: Input DataFrame
            method: 'cap' (winsorization) or 'remove'
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        print(f"\n🔧 Handling Outliers (method: {method})...")
        
        for col in numerical_cols:
            outliers = self.detect_outliers_iqr(df_clean, col)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                if method == 'cap':
                    # Winsorization: cap at 1st and 99th percentile
                    lower = df_clean[col].quantile(0.01)
                    upper = df_clean[col].quantile(0.99)
                    df_clean[col] = df_clean[col].clip(lower, upper)
                    print(f"   • {col}: Capped {n_outliers} outliers")
                elif method == 'remove':
                    df_clean = df_clean[~outliers]
                    print(f"   • {col}: Removed {n_outliers} outlier rows")
        
        return df_clean
    
    def convert_date_columns(self, df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
        """
        Convert columns to datetime format
        
        Args:
            df: Input DataFrame
            date_columns: List of column names to convert
            
        Returns:
            DataFrame with converted date columns
        """
        df_clean = df.copy()
        
        print(f"\n🔧 Converting Date Columns...")
        
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col])
                print(f"   • {col}: Converted to datetime")
        
        return df_clean
    
    def preprocess(self, df: pd.DataFrame, 
                create_date_column: bool = True,
                outlier_method: str = 'cap') -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            create_date_column: Whether to create Date column from Year/Month
            outlier_method: Method for handling outliers
            
        Returns:
            Cleaned DataFrame
        """
        print("\n" + "="*80)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Remove duplicates
        df_clean = self.remove_duplicates(df)
        
        # Step 2: Create Date column from Year and Month
        if create_date_column and 'Year' in df_clean.columns and 'Month' in df_clean.columns:
            print(f"\n🔧 Creating Date column from Year and Month...")
            df_clean['Date'] = pd.to_datetime(df_clean[['Year', 'Month']].assign(day=1))
            print(f"   • Date column created")
        
        # Step 3: Handle missing values
        df_clean = self.handle_missing_values(df_clean)
        
        # Step 4: Handle outliers
        df_clean = self.handle_outliers(df_clean, method=outlier_method)
        
        self.cleaned_shape = df_clean.shape
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape:  {self.cleaned_shape}")
        print(f"Rows removed:   {self.original_shape[0] - self.cleaned_shape[0]}")
        print("="*80 + "\n")
        
        return df_clean

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(random_state=42)
    
    # Load data
    df = preprocessor.load_data('data/raw/tesla_data.csv')
    
    # Print summary
    preprocessor.print_data_summary(df)
    
    # Preprocess
    df_clean = preprocessor.preprocess(df, date_columns=['Date'])
    
    # Save cleaned data
    output_path = 'data/processed/tesla_clean.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"✓ Cleaned data saved to {output_path}")