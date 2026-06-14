"""
Classification models and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import streamlit as st

@st.cache_resource
def train_all_models(X_train, X_test, y_train, y_test):
    """Train all classification models"""
    models = {}
    predictions = {}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    predictions['Logistic Regression'] = lr.predict(X_test)
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt
    predictions['Decision Tree'] = dt.predict(X_test)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    predictions['Random Forest'] = rf.predict(X_test)
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=100, random_state=42, max_depth=5, 
                        eval_metric='mlogloss', use_label_encoder=False)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    predictions['XGBoost'] = xgb.predict(X_test)
    
    return models, predictions

def evaluate_models(y_test, predictions):
    """Evaluate all models"""
    results = []
    
    for model_name, y_pred in predictions.items():
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted')
        }
        results.append(metrics)
    
    return pd.DataFrame(results)

def get_feature_importance(model, feature_names):
    """Get feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        return importance
    return None

def split_data(X, y, test_size=0.25, random_state=42):
    """Split data into train and test sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)