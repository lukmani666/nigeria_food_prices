import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AdvancePricePredictor:
    """Advance price predict with multiple ML algorithms"""
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'xgboost': xgb.XGBRegressor(randome_state=42),
            'lightgbm': lgb.LGBMRegressor(random_state=42)
        }
        self.fitted_models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.performance_metrics = {}
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for price prediction"""
        features_df = df.copy()

        #ensure datetime index
        if 'date' in features_df.columns:
            features_df['date'] = pd.to_datetime(features_df['date'])
            features_df.set_index('date', inplace=True)
        
        #time features
        features_df['year'] = features_df.index.year
        features_df['month'] = features_df.index.month
        features_df['quarter'] = features_df.index.quarter
        features_df['day_of_year'] = features_df.index.dayofyear
        features_df['week_of_year'] = features_df.index.isocalendar().week

        #cyclical encoding
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)

        #lag features (grouped by commodity and market)
        for lag in [1, 2, 3, 6, 12]:
            features_df[f'price_lag_{lag}'] = features_df.groupby(['commodity', 'market'])['price'].shift(lag)
        
        #rolling statistics (grouped by commodity and market)
        for window in [3, 6, 12]:
            grouped = features_df.groupby(['commodity', 'market'])['price']
            features_df[f'price_ma_{window}'] = grouped.rolling(window=window).mean().reset_index(0, drop=True)
            features_df[f'price_std_{window}'] = grouped.rolling(window=window).std().reset_index(0, drop=True)
            features_df[f'price_min_{window}'] = grouped.rolling(window=window).min().reset_index(0, drop=True)
            features_df[f'price_max_{window}'] = grouped.rolling(window=window).max().reset_index(0, drop=True)
        
        #market-level features
        market_stats = features_df.groupby('market').agg({
            'price': ['mean', 'std'],
            'commodity': 'nunique'
        })
        market_stats.columns = ['market_avg_price', 'market_price_std', 'market_commodity_count']
        features_df = features_df.join(market_stats, on='market')

        #commodity-level features
        commodity_stats = features_df.groupby('commodity').agg({
            'price': ['mean', 'std'],
            'market': 'nunique'
        })
        commodity_stats.columns = ['commodity_avg_price', 'commodity_price_std', 'commodity_market_count']
        features_df = features_df.join(commodity_stats, on='commodity')

        #state-level features
        state_stats = features_df.groupby('admin1').agg({
            'price': ['mean', 'std']
        })
        state_stats.columns = ['state_avg_price', 'state_price_std']
        features_df = features_df.join(state_stats, on='admin1')

        #price ratios
        features_df['price_to_market_avg'] = features_df['price'] / features_df['market_avg_price']
        features_df['price_to_commodity_avg'] = features_df['price'] / features_df['commodity_avg_price']
        features_df['price_to_state_avg'] = features_df['price'] / features_df['state_avg_price']

        return features_df.reset_index(drop=True)

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df_encoded = df.copy()
        categorical_cols = ['commodity', 'market', 'admin1', 'admin2']

        for col in categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    if col in self.encoders:
                        #handle unseen categories
                        df_encoded[f'{col}_encoded'] = df_encoded[col].astype(str).apply(
                            lambda x: self.encoders[col].transform([x])[0] if x in self.encoders[col].classes_ else -1
                        )
        
        return df_encoded
    
    def train_model(
        self, df: pd.DataFrame, model_name: str = 'xgboost',
        target_col: str = 'price', test_size: float = 0.2,
        optimize_hyperparams: bool = False
    ) -> Dict[str, Any]:
        """Train individual model"""
        try:
            #prepare features
            features_df = self.prepare_features(df)
            features_df = self.encode_categorical_features(features_df, fit=True)

            #remove rows with Nan values
            features_df = features_df.dropna()

            if len(features_df) < 100:
                raise ValueError("Insufficient data for training after feature engineering")
            
            #define features columns
            features_cols = [col for col in features_df.columns if col not in [target_col, 'commodity', 'market', 'admin1', 'admin2', 'usdprice']]

            X = features_df[features_cols]
            y = features_df[target_col]

            #time-based split
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            #scale features for certain models
            if model_name in ['gradient_boosting']:
                self.scalers[model_name] = StandardScaler()
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            #get model
            model = self.models[model_name]

            #hyperparameter optimization
            if optimize_hyperparams:
                param_grids = {
                    'random_forest': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5]
                    },
                    'xgboost': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2]
                    },
                    'lightgbm': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2]
                    }
                }

                if model_name in param_grids:
                    tscv = TimeSeriesSplit(n_splits=3)
                    grid_search = GridSearchCV(
                        model, param_grids[model_name],
                        cv=tscv, scoring='neg_mean_absolute_error',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    model = grid_search.best_estimator_
                    logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            
            #train model
            model.fit(X_train_scaled, y_train)
            self.fitted_models[model_name] = model

            #make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            #calculate metrics
            metrics = {
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_mape': np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100,
                'test_mape': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            }
            self.performance_metrics[model_name] = metrics

            #feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': features_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[model_name] = importance_df
            
            logger.info(f"{model_name} training complete. Test MAE: {metrics['test_mae']:.2f}")

            return {
                'model': model,
                'metrics': metrics,
                'feature_importance': self.feature_importance.get(model_name),
                'predictions': {
                    'train': {'actual': y_train, 'predicted': y_pred_train},
                    'test': {'actual': y_test, 'predicted': y_pred_test}
                }
            }
        except Exception as e:
            logger.error(f'Error training {model_name}: {e}')
            raise
    
    def train_ensemble(self, df: pd.DataFrame, models: List[str] = None) -> Dict[str, Any]:
        """Train ensamble of models"""
        if models is None:
            models = ['random_forest', 'xgboost', 'lightgbm']
        
        ensemble_results = {}

        for model_name in models:
            if model_name in self.models:
                try:
                    result = self.train_model(df, model_name)
                    ensemble_results['model_name'] = result
                except Exception as e:
                    logger.warning(f'Failed to train {model_name}: {e}')
        
        #create ensamble predictions (simple average)
        if len(ensemble_results) > 1:
            test_predictions = []
            for model_name, result in ensemble_results.items():
                test_predictions.append(result['predictions']['test']['predicted'])
            
            ensemble_pred = np.mean(test_predictions, axis=0)
            actual_values = ensemble_results[list(ensemble_results.keys())[0]]['predictions']['test']['actual']

            ensemble_mae = mean_absolute_error(actual_values, ensemble_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(actual_values, ensemble_pred))
            ensemble_mape = np.mean(np.abs((actual_values - ensemble_pred) / actual_values)) * 100

            ensemble_results['ensemble'] = {
                'metrics': {
                    'test_mae': ensemble_mae,
                    'test_rmse': ensemble_rmse,
                    'test_mape': ensemble_mape
                },
                'predictions': {
                    'test': {'actual': actual_values, 'predicted': ensemble_pred}
                }
            }
            logger.info(f'Ensemble training complete. Test MAE: {ensemble_mae:.2f}')

        return ensemble_results
    
    def predict_prices(self, df: pd.DataFrame, model_name: str = 'ensemble') -> np.ndarray:
        """Make predictions using trained model"""
        if model_name not in self.fitted_models and model_name != 'ensemble':
            raise ValueError(f'Model {model_name} not trained')
        
        #prepare features
        features_df = self.prepare_features(df)
        features_df = self.encode_categorical_features(features_df, fit=False)

        #remove target column if present
        features_cols = [col for col in features_df.columns
                         if col not in ['price', 'commodity', 'market', 'admin1', 'admin2', 'usdprice']]
        
        X = features_df[features_cols].fillna(0)

        if model_name == 'ensemble' and len(self.fitted_models) > 1:
            #ensemble prediction
            predictions = []
            for name, model in self.fitted_models.items():
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
                else:
                    X_scaled = X
                pred = model.predict(X_scaled)
                predictions.append(pred)
            
            return np.mean(predictions, axis=0)
        else:
            #single model prediction
            model = self.fitted_models[model_name]
            if model_name in self.scalers:
                X_scaled = self.scalers[model_name].transform(X)
            else: 
                X_scaled = X
            return model.predict(X_scaled)
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Compare performance of all trained models"""
        comparison_data = []

        for model_name, metrics in self.performance_metrics.items():
            comparison_data.append({
                'Model': model_name,
                'Train MAE': metrics['train_mae'],
                'Test MAE': metrics['test_mae'],
                'Train RMSE': metrics['train_rmse'],
                'Test RMSE': metrics['test_rmse'],
                'Train MAPE': metrics['train_mape'],
                'Test MAPE': metrics['test_mape']
            })
        return pd.DataFrame(comparison_data).round(2)


