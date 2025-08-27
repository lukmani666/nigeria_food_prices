import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """Comprehensive time series analysis for food price data"""

    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.features_created = []
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime"""
        df_features = df.copy()

        #ensure datetime index
        if not isinstance(df_features.index, pd.DatetimeIndex):
            df_features['date'] = pd.to_datetime(df_features['date'])
            df_features.set_index('date', inplace=True)
        
        #time features
        df_features['year'] = df_features.index.year
        df_features['month'] = df_features.index.month
        df_features['quarter'] = df_features.index.quarter
        df_features['day_of_year'] = df_features.index.dayofyear
        df_features['week_of_year'] = df_features.index.isocalendar().week

        #cyclical features
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
        df_features['quarter_cos'] = np.sin(2 * np.pi * df_features['quarter'] / 4)

        #time trend
        df_features['time_trend'] = range(len(df_features))

        self.features_created = ['year', 'month', 'quarter', 'day_of_year', 'week_of_year', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 'time_trend']

        return df_features
    
    def create_lag_features(self, df: pd.DataFrame, price_col: str = 'price', lags: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
        """Create lagged price features"""
        df_lg = df.copy()

        for lag in lags:
            df_lg[f'price_lag_{lag}'] = df_lg[price_col].shift(lag)
            self.features_created.append(f'price_lag_{lag}')
        
        return df_lg
    
    def create_rolling_features(self, df:pd.DataFrame, price_col: str = 'price', windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """Create rolling statistical features"""
        df_rolling = df.copy()

        for window in windows:
            #rolling mean
            df_rolling[f'price_ma_{window}'] = df_rolling[price_col].rolling(window=window).mean()

            #rolling standard deviation
            df_rolling[f'price_std_{window}'] = df_rolling[price_col].rolling(window=window).std()

            #rolling min/max
            df_rolling[f'price_min_{window}'] = df_rolling[price_col].rolling(window=window).min()
            df_rolling[f'price_max_{window}'] = df_rolling[price_col].rolling(window=window).max()

            #price relative to moving average
            df_rolling[f'price_ma_ratio_{window}'] = df_rolling[price_col] / df_rolling[f'price_ma_{window}']

            #rolling volatility
            returns = df_rolling[price_col].pct_change()
            df_rolling[f'volatility_{window}'] = returns.rolling(window=window).std()

            self.features_created.append([
                f'price_ma_{window}', f'price_std_{window}',
                f'price_min_{window}', f'price_max_{window}',
                f'price_ma_ratio_{window}', f'volatility_{window}'
            ])
        
        return df_rolling
    
    def create_all_features(self, df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """Create comprehensive feature set"""
        #set with original data
        df_features = df.copy()

        #time features
        df_features = self.create_time_features(df_features)

        #lag features
        df_features =self.create_lag_features(df_features, price_col)

        #rolling features
        df_features = self.create_rolling_features(df_features, price_col)

        #drop rows with nan values created by lags and rolling windows
        df_features = df_features.dropna()

        return df_features

class PriceForestModel:
    """Advance price forecasting model with multiple algorithms"""
    def __init__(self, model_type: str = 'gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.analyzer = TimeSeriesAnalyzer()
        self.features_columns = []
        self.is_fitted = False
        self.performance_metrics = {}

        #initialize model based on type
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.model = self.models.get(model_type, self.models['gradient_boosting'])
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'price') -> Tuple[pd.DataFrame, pd.Series]:
        """prepare data for modeling"""
        #create features
        df_features = self.analyzer.create_all_features(df, target_col)

        #define feature columns (exclude target)
        self.features_columns = [col for col in df_features.columns if col != target_col]

        X = df_features[self.features_columns]
        y = df_features[target_col]

        return X, y
    
    def train(self, df: pd.DataFrame, target_col: str = 'price', test_size: float = 0.2, validate: bool = True) -> Dict[str, Any]:
        """Train the forecasting model"""
        try:
            #prepare data
            X, y = self.prepare_data(df, target_col)
            
            if len(X) < 20:
                raise ValueError("Insufficient data for training. Need at least 20 samples.")
            
            #time series split for validation
            if validate and len(X) > 50:
                tscv = TimeSeriesSplit(n_splits=3)
                cv_score = cross_val_score(self.model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
                logger.info(f'Cross-validation MAE: {-cv_score.mean():.2f} ± {cv_score.std:.2f}')
            
            #split data for final evaluation
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            #scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            #train model
            self.model.fit(X_train_scaled, y_train)

            #make predictions
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)

            #calculate metrics
            self.performance_metrics = {
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train)),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_mape': np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100,
                'test_mape': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            }

            #feature importance (if available)
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.features_columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)

                self.performance_metrics['feature_importance'] = feature_importance
            self.is_fitted = True
            logger.info(f'Model training complete. Test MAE: {self.performance_metrics['test_mae']:.2f}')

            return {
                'performance_metrics': self.performance_metrics,
                'predictions': {
                    'train': {'actual': y_train, 'predicted': y_pred_train},
                    'test': {'actual': y_test, 'predicted': y_pred_test}
                }
            }
        except Exception as e:
            logger.error(f'Error in model training: {e}')
            raise
    
    def predict(self, periods: int, last_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate forecasts for future periods"""
        if not self.is_fitted:
            raise ValueError("Model must be trained befor making predicts")
        
        if last_data is None:
            raise ValueError("Need recent data to generate forecast")
        
        try:
            predictions = []
            confidence_intervals = []

            #use the last available data point as starting point
            current_data = last_data.copy()

            for period in range(periods):
                #create features for current data point
                X_current, _ = self.prepare_data(current_data)
                
                if len(X_current) == 0:
                    break

                #use the last row for prediction
                X_pred = X_current.iloc[[-1]][self.features_columns]
                X_pred_scaled = self.scaler.transform(X_pred)

                #make prediction
                prediction = self.model.predict(X_pred_scaled)[0]
                predictions.append(prediction)

                #simple confidence interval estimation
                #(in practice, you'd use more sophisticated methods)
                prediction_std = self.performance_metrics.get('test_rmse', prediction * 0.1)
                ci_lower = prediction - 1.96 * prediction_std
                ci_upper = prediction + 1.96 * prediction_std
                confidence_intervals.append((ci_lower, ci_upper))

                #update data for next prediction
                #this is simplified - in practice, you'd need proper time indexing
                new_row = current_data.iloc[[-1]].copy()
                new_row['price'] = prediction
                current_data = pd.concat([current_data, new_row])

            return {
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'model_type': self.model_type,
                'performance': self.performance_metrics
            }
        except Exception as e:
            logger.error(f'Error in prediction: {e}')
            raise
    
    def save_model(self, filepath: Path):
        """Save trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.features_columns,
            'model_type': self.model_type,
            'performance_metrics': self.performance_metrics,
            'analyzer': self.analyzer
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load trained model"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.features_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.performance_metrics = model_data['performance_metrics']
        self.analyzer = model_data['analyzer']
        self.is_fitted = True

        logger.info(f"Model loaded from {filepath}")