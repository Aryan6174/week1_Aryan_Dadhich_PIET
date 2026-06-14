"""
Data loading and inspection utilities
"""

import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    """Load data from file path or uploaded file"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif file_path is not None:
            df = pd.read_csv(file_path)
        else:
            raise ValueError("No data source provided")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_data_info(df):
    """Get comprehensive data information"""
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    return info

def get_summary_stats(df):
    """Get summary statistics"""
    return df.describe()

def check_data_quality(df):
    """Check data quality issues"""
    issues = []
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        issues.append(f"Missing values found: {missing.sum()} total")
    
    # Check duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Duplicate rows found: {duplicates}")
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isin([float('inf'), float('-inf')]).any():
            issues.append(f"Infinite values found in {col}")
    
    if not issues:
        issues.append("✓ No data quality issues detected")
    
    return issues