import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """Handles cleaning and preprocessing of the Nigerian Food Prices dataset"""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log = []

    def clean_all(self,
                  remove_duplicates: bool = True,
                  handle_missing_prices: str = 'drop',
                  handle_invalid_coords: str = 'drop',
                  handle_outliers: str = 'clip',
                  outlier_method: str = 'iqr') -> pd.DataFrame:
        """
        Comprehensive data cleaning pipeline
        
        Args:
            remove_duplicates: Whether to remove duplicate records
            handle_missing_prices: 'drop', 'interpolate', or 'keep'
            handle_invalid_coords: 'drop', 'impute', or 'keep'  
            handle_outliers: 'drop', 'clip', or 'keep'
            outlier_method: 'iqr', 'zscore', or 'percentile'
        """
        logger.info("Starting comprehensive data cleaning...")
        if remove_duplicates:
            self.remove_duplicates()
        
        self.clean_dates()
        self.clean_prices(method=handle_missing_prices)
        self.clean_coordinates(method=handle_invalid_coords)
        self.clean_categorical_data()

        if handle_outliers != 'keep':
            self.handle_price_outliers(method=handle_outliers, detection=outlier_method)
        
        self.standardize_units()
        self._log_cleaning_summary()

        return self.df
    
    def remove_duplicates(self) -> None:
        """Remove duplicate records"""
        initial_count = len(self.df)

        # Define key columns for duplicate detection
        key_columns = ['date', 'market', 'commodity', 'price']
        self.df = self.df.drop_duplicates(subset=key_columns, keep='first')

        removed_count = initial_count - len(self.df)
        self.cleaning_log.append(f"Removed {removed_count:,} duplicate record")
        logger.info(f"Reward {removed_count:,} duplicates")
    
    def clean_dates(self) -> None:
        """Clean and validate date information"""
        initial_count = len(self.df)

        #Remove records with invalid dates
        self.df = self.df.dropna(subset=['date'])

        #Add time components
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['day_of_year'] = self.df['date'].dt.dayofyear

        removed_count = initial_count - len(self.df)
        if removed_count > 0:
            self.cleaning_log.append(f"Removed {removed_count:,} records with invalid dates")
    
    def clean_prices(self, method: str = 'drop') -> None:
        """Clean price data"""
        initial_count = len(self.df)

        if method == 'drop':
            # Remove records with missing, zero, or negative prices
            self.df = self.df[
                (self.df['price'] > 0) &
                (self.df['usdprice'] > 0) &
                (self.df['price'].notna()) &
                (self.df['usdprice'].notna())
            ]
        elif method == 'interpolate':
            # Interpolate missing prices by commodity and market
            self.df['price'] = self.df.groupby(['commodity', 'market'])['price'].transform(
                lambda x: x.interpolate(method='linear')
            )

            self.df['usdprice'] = self.df.groupby(['commodity', 'market'])['usdprice'].transform(
                lambda x: x.interpolate(method='linear')
            )

            # Still remove zeros and negatives
            self.df = self.df[(self.df['price'] > 0) & (self.df['usdprice'] > 0)]
        
        removed_count = initial_count - len(self.df)
        if removed_count > 0:
            self.cleaning_log.append(f"Cleaned {removed_count:,} records with invalid prices")
    

    def clean_coordinates(self, method: str = 'drop') -> None:
        """Clean geographic coordinates"""
        initial_count = len(self.df)

        # Nigeria approximate bounds
        lat_bounds = (4, 14)
        lon_bounds = (3, 16)

        if method == 'drop':
            # Remove records with invalid coordinates
            valid_coords = (
                (self.df['latitude'].between(lat_bounds[0], lat_bounds[1])) &
                (self.df['longitude'].between(lon_bounds[0], lon_bounds[1])) &
                (self.df['latitude'].notna()) &
                (self.df['longitude'].notna())
            )
            self.df = self.df[valid_coords]
        elif method == 'impute': 
            # Impute missing coordinates with market averages
            market_coords = self.df.groupby('market')[['latitude', 'longitude']].mean()
            self.df['latitude'] = self.df['latitude'].fillna(
                self.df['market'].map(market_coords['latitude'])
            )
            self.df['longitude'] = self.df['longitude'].fillna(
                self.df['market'].map(market_coords['longitude'])
            )
        
        removed_count = initial_count - len(self.df)
        if removed_count > 0:
            self.cleaning_log.append(f"Cleaned {removed_count:,} records with invalid coordinates")
    
    def clean_categorical_data(self) -> None:
        """Clean and standardize categorical variables"""
        # Standardize string columns
        string_columns = ['admin1', 'admin2', 'market', 'commodity', 'category']

        for col in string_columns:
            if col in self.df.columns:
                # Strip whitespace and standardize case
                self.df[col] = self.df[col].astype(str).str.strip().str.title()

                # Remove records with missing categorical data
                self.df = self.df[self.df[col].notna()]
        self.cleaning_log.append("Standardized categorical variables")
    
    def handle_price_outliers(self, method: str = 'clip', detection: str = 'iqr') -> None:
        """Handle price outliers"""
        initial_count = len(self.df)

        for price_col in ['price', 'usdprice']:
            if detection == 'iqr':
                Q1 = self.df[price_col].quantile(0.25)
                Q3 = self.df[price_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q1 + 3 * IQR
            elif detection == 'zscore':
                mean = self.df[price_col].mean()
                std = self.df[price_col].std()
                lower_bound = mean - 4 * std
                upper_bound = mean + 4 * std
            elif detection == 'percentile':
                lower_bound = self.df[price_col].quantile(0.01)
                upper_bound = self.df[price_col].quantile(0.99)
            
            if method == 'clip':
                self.df[price_col] = self.df[price_col].clip(lower_bound, upper_bound)
            elif method == 'drop':
                self.df = self.df[
                    (self.df[price_col] >= lower_bound) &
                    (self.df[price_col] <= upper_bound)
                ]
        
        removed_count = initial_count - len(self.df)
        if removed_count > 0:
            self.cleaning_log.append(f"Handled {removed_count:,} outlier records")
    
    def standardize_units(self) -> None:
        """Standardize unit measurements"""
        if 'unit' in self.df.columns:
            #common unit standardizations
            unit_mapping = {
                'kg': 'kg',
                'kilogram': 'kg',
                'kilograms': 'kg',
                'gram': 'g',
                'grams': 'g',
                'litre': 'l',
                'liter': 'l',
                'litres': 'l',
                'liters': 'l'
            }
            self.df['unit_standardized'] = self.df['unit'].str.lower().map(unit_mapping).fillna(self.df['unit'])
            self.cleaning_log.append("Standardized unit measurements")
    
    def _log_cleaning_summary(self) -> None:
        """Log cleaning summary"""
        final_shape = self.df.shape
        total_removed = self.original_shape[0] - final_shape[0]
        retention_rate = (final_shape[0] / self.original_shape[0] * 100)

        logger.info(f"Cleaning complete:")
        logger.info(f"  Original shape: {self.original_shape}")
        logger.info(f"  Final shape: {final_shape}")
        logger.info(f"  Records removed: {total_removed:,}")
        logger.info(f"  Retention rate: {retention_rate:.1f}%")

        self.cleaning_log.append(f"Final retention rate: {retention_rate:.1f}%")

    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get comprehensive cleaning report"""
        return {
            'original_shape': self.original_shape,
            'final_shape': self.df.shape,
            'cleaning_steps': self.cleaning_log,
            'retention_rate': (self.df.shape[0] / self.original_shape[0]) * 100,
            'final_data_quality': self._assess_final_quality()
        }
    
    def _assess_final_quality(self) -> Dict[str, Any]:
        """Assess final data quality after cleaning"""
        return {
            'missing_data': self.df.isnull().sum().to_dict(),
            'price_stats': {
                'ngn_min': self.df['price'].min(),
                'ngn_max': self.df['price'].max(),
                'ngn_mean': self.df['price'].mean(),
                'usd_min': self.df['usdprice'].min(),
                'usd_max': self.df['usdprice'].max(),
                'usd_mean': self.df['usdprice'].mean()
            },
            'coverage': {
                'states': self.df['admin1'].nunique(),
                'markets': self.df['market'].nunique(),
                'commodities': self.df['commodity'].nunique(),
                'date_range': (self.df['date'].min(), self.df['date'].max())
            }
        }


