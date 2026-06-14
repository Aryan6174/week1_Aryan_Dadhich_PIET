"""
Data preprocessing utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st

def preprocess_data(df, target_col='country'):
    """Preprocess data for modeling"""
    # Store country names
    countries = df[target_col].copy() if target_col in df.columns else None
    
    # Remove non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    return df_numeric, countries

@st.cache_data
def scale_features(df):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return scaled_df, scaler

@st.cache_data
def apply_pca(scaled_data, n_components=2):
    """Apply PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(scaled_data)
    
    pca_df = pd.DataFrame(
        data=pca_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    explained_variance = pca.explained_variance_ratio_
    
    return pca_df, explained_variance, pca

def handle_missing_values(df, strategy='mean'):
    """Handle missing values"""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'drop':
        return df.dropna()
    else:
        return df

def detect_outliers(df, method='iqr', threshold=1.5):
    """Detect outliers using IQR method"""
    outliers = pd.DataFrame(index=df.index)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    return outliers