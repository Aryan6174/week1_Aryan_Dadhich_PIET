# src/train.py
"""
Model Training Module for Tesla ML Pipeline
Implements multiple regression models with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Comprehensive model training and evaluation class
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    
    def prepare_data(self, df: pd.DataFrame,
                 target_col: str = 'Estimated_Deliveries',  # UPDATED
                 feature_cols: Optional[List[str]] = None,
                 test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            target_col: Target variable column name (default: 'Estimated_Deliveries')
            feature_cols: List of feature columns (if None, use all except target)
            test_size: Proportion of test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        print("\n" + "="*80)
        print("PREPARING DATA FOR TRAINING")
        print("="*80)
        
        df_model = df.copy()
        
        # Remove date columns and non-numeric columns
        date_cols = df_model.select_dtypes(include=['datetime64']).columns.tolist()
        df_model = df_model.drop(columns=date_cols, errors='ignore')
        
        # Remove categorical columns (or encode them)
        categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print(f"\n🔧 Encoding categorical features: {categorical_cols}")
            df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
        
        # Prepare features and target
        if feature_cols is None:
            feature_cols = [col for col in df_model.columns if col != target_col]
        
        missing_cols = [col for col in feature_cols if col not in df_model.columns]

        if missing_cols:
            print("Missing columns removed:", missing_cols)
            feature_cols = [col for col in feature_cols if col in df_model.columns]

        X = df_model[feature_cols]
        y = df_model[target_col]
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to keep column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print(f"\n✓ Data prepared successfully!")
        print(f"   • Features: {len(feature_cols)}")
        print(f"   • Training samples: {len(X_train_scaled)}")
        print(f"   • Test samples: {len(X_test_scaled)}")
        print(f"   • Target: {target_col}")
        print("="*80 + "\n")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train Linear Regression model"""
        print("\n📊 Training Linear Regression...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        self.models['Linear Regression'] = model
        
        result = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_rmse': cv_rmse,
            'predictions': {'train': y_train_pred, 'test': y_test_pred}
        }
        
        print(f"   ✓ Test RMSE: {test_metrics['RMSE']:,.2f}")
        print(f"   ✓ Test R²: {test_metrics['R2']:.4f}")
        print(f"   ✓ CV RMSE: {cv_rmse:,.2f}")
        
        return result
    
    def train_ridge(self, X_train, X_test, y_train, y_test, alpha: float = 1.0) -> Dict:
        """Train Ridge Regression model"""
        print("\n📊 Training Ridge Regression...")
        
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        self.models['Ridge'] = model
        
        result = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_rmse': cv_rmse,
            'predictions': {'train': y_train_pred, 'test': y_test_pred}
        }
        
        print(f"   ✓ Test RMSE: {test_metrics['RMSE']:,.2f}")
        print(f"   ✓ Test R²: {test_metrics['R2']:.4f}")
        print(f"   ✓ CV RMSE: {cv_rmse:,.2f}")
        
        return result
    
    def train_lasso(self, X_train, X_test, y_train, y_test, alpha: float = 1.0) -> Dict:
        """Train Lasso Regression model"""
        print("\n📊 Training Lasso Regression...")
        
        model = Lasso(alpha=alpha, random_state=self.random_state, max_iter=10000)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        self.models['Lasso'] = model
        
        # Count non-zero coefficients
        n_features_used = np.sum(model.coef_ != 0)
        
        result = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_rmse': cv_rmse,
            'predictions': {'train': y_train_pred, 'test': y_test_pred},
            'n_features_used': n_features_used
        }
        
        print(f"   ✓ Test RMSE: {test_metrics['RMSE']:,.2f}")
        print(f"   ✓ Test R²: {test_metrics['R2']:.4f}")
        print(f"   ✓ CV RMSE: {cv_rmse:,.2f}")
        print(f"   ✓ Features used: {n_features_used}/{len(X_train.columns)}")
        
        return result
    
    def train_random_forest(self, X_train, X_test, y_train, y_test,
                           n_estimators: int = 100,
                           max_depth: Optional[int] = None) -> Dict:
        """Train Random Forest model"""
        print("\n📊 Training Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        self.models['Random Forest'] = model
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        result = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_rmse': cv_rmse,
            'predictions': {'train': y_train_pred, 'test': y_test_pred},
            'feature_importance': feature_importance
        }
        
        print(f"   ✓ Test RMSE: {test_metrics['RMSE']:,.2f}")
        print(f"   ✓ Test R²: {test_metrics['R2']:.4f}")
        print(f"   ✓ CV RMSE: {cv_rmse:,.2f}")
        
        return result
    
    def train_xgboost(self, X_train, X_test, y_train, y_test,
                     n_estimators: int = 100,
                     learning_rate: float = 0.1,
                     max_depth: int = 6) -> Dict:
        """Train XGBoost model"""
        print("\n📊 Training XGBoost...")
        
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        self.models['XGBoost'] = model
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        result = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_rmse': cv_rmse,
            'predictions': {'train': y_train_pred, 'test': y_test_pred},
            'feature_importance': feature_importance
        }
        
        print(f"   ✓ Test RMSE: {test_metrics['RMSE']:,.2f}")
        print(f"   ✓ Test R²: {test_metrics['R2']:.4f}")
        print(f"   ✓ CV RMSE: {cv_rmse:,.2f}")
        
        return result
    
    def train_all_models(self, X_train, X_test, y_train, y_test) -> Dict:
        """
        Train all models and compare results
        
        Returns:
            Dictionary containing all model results
        """
        print("\n" + "="*80)
        print("TRAINING ALL MODELS")
        print("="*80)
        
        results = {}
        
        # Train each model
        results['Linear Regression'] = self.train_linear_regression(X_train, X_test, y_train, y_test)
        results['Ridge'] = self.train_ridge(X_train, X_test, y_train, y_test)
        results['Lasso'] = self.train_lasso(X_train, X_test, y_train, y_test)
        results['Random Forest'] = self.train_random_forest(X_train, X_test, y_train, y_test)
        results['XGBoost'] = self.train_xgboost(X_train, X_test, y_train, y_test)
        
        self.results = results
        
        # Find best model
        best_model_name = min(results.keys(), 
                             key=lambda x: results[x]['test_metrics']['RMSE'])
        self.best_model = results[best_model_name]['model']
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Test RMSE: {results[best_model_name]['test_metrics']['RMSE']:,.2f}")
        print(f"   Test R²: {results[best_model_name]['test_metrics']['R2']:.4f}")
        print("="*80 + "\n")
        
        return results
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table of all models
        
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train_MAE': result['train_metrics']['MAE'],
                'Test_MAE': result['test_metrics']['MAE'],
                'Train_RMSE': result['train_metrics']['RMSE'],
                'Test_RMSE': result['test_metrics']['RMSE'],
                'Train_R2': result['train_metrics']['R2'],
                'Test_R2': result['test_metrics']['R2'],
                'Test_MAPE': result['test_metrics']['MAPE'],
                'CV_RMSE': result['cv_rmse']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.round(4)
        
        return df_comparison
    
    def tune_random_forest(self, X_train, y_train, method: str = 'random') -> Dict:
        """
        Hyperparameter tuning for Random Forest
        
        Args:
            X_train: Training features
            y_train: Training target
            method: 'grid' or 'random'
            
        Returns:
            Dictionary with best parameters and score
        """
        print(f"\n🔧 Tuning Random Forest using {method.upper()}SearchCV...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        if method == 'grid':
            search = GridSearchCV(
                rf, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                rf, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1, n_iter=20, random_state=self.random_state
            )
        
        search.fit(X_train, y_train)
        
        print(f"\n✓ Tuning complete!")
        print(f"   Best parameters: {search.best_params_}")
        print(f"   Best CV RMSE: {np.sqrt(-search.best_score_):,.2f}")
        
        return {
            'best_params': search.best_params_,
            'best_score': np.sqrt(-search.best_score_),
            'best_model': search.best_estimator_
        }
    
    def tune_xgboost(self, X_train, y_train, method: str = 'random') -> Dict:
        """
        Hyperparameter tuning for XGBoost
        
        Args:
            X_train: Training features
            y_train: Training target
            method: 'grid' or 'random'
            
        Returns:
            Dictionary with best parameters and score
        """
        print(f"\n🔧 Tuning XGBoost using {method.upper()}SearchCV...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 6, 9],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }
        
        xgb = XGBRegressor(random_state=self.random_state, n_jobs=-1)
        
        if method == 'grid':
            search = GridSearchCV(
                xgb, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                xgb, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1, n_iter=20, random_state=self.random_state
            )
        
        search.fit(X_train, y_train)
        
        print(f"\n✓ Tuning complete!")
        print(f"   Best parameters: {search.best_params_}")
        print(f"   Best CV RMSE: {np.sqrt(-search.best_score_):,.2f}")
        
        return {
            'best_params': search.best_params_,
            'best_score': np.sqrt(-search.best_score_),
            'best_model': search.best_estimator_
        }
    
    def save_model(self, model, model_name: str, save_dir: str = 'models/saved_models'):
        """
        Save trained model
        
        Args:
            model: Trained model object
            model_name: Name for saving
            save_dir: Directory to save model
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_dir) / f'{model_name}.joblib'
        joblib.dump(model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, model_name: str, load_dir: str = 'models/saved_models'):
        """
        Load saved model
        
        Args:
            model_name: Name of model to load
            load_dir: Directory containing model
            
        Returns:
            Loaded model object
        """
        filepath = Path(load_dir) / f'{model_name}.joblib'
        model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Example usage
    print("Model Trainer module loaded successfully!")