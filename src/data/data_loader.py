import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and basic validation of the Nigerian Food Prices dataset"""
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.df = None
        self.metadata = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load the main dataset with basic validation"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path, skiprows=[1])

            #basic validation
            self._validate_required_columns()
            self._convert_data_types()
            self._extract_metadata()

            logger.info(f"Successfully loaded data with shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_required_columns(self):
        """Ensure required columns are present"""
        required_cols = ['date', 'admin1', 'admin2', 'market', 'commodity', 'price', 'usdprice', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _convert_data_types(self):
        """Convert columns to appropriate data types"""
        try:
            self.df['date'] = pd.to_datetime(self.df['date'])

            #convert numeric columns
            numeric_cols = ['price', 'usdprice', 'latitude', 'longitude']
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            #covert categorical columns
            categorical_cols = ['admin1', 'admin2', 'market', 'commodity', 'category', 'unit']
            for col in categorical_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype('category')
        except Exception as e:
            logger.warning(f"Error in data type convertion: {e}")
    
    def _extract_metadata(self):
        """Extract basic metadata about the dataset"""
        if self.df is None:
            return
        
        self.metadata = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'date_range': (self.df['date'].min(), self.df['date'].max()),
            'unique_commodities': self.df['commodity'].nunique(),
            'unique_admin1': self.df['admin1'].nunique(),
            'unique_admin2': self.df['admin2'].nunique(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict()
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata"""
        return self.metadata
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return {
            'basic_stats': self.df.describe(),
            'categorical_stats': self.df.describe(include=["category"]),
            'missing_data': self.df.isnull().sum(),
            'data_quality': self._assess_data_quality()
        }
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess basic data quality metrics"""
        quality_metrics = {}

        #price qaulity
        quality_metrics['valid_prices'] = {
            'ngn_positive': (self.df['price'] > 0).sum(),
            'usd_positive': (self.df['usdprice'] > 0).sum(),
            'ngn_zero_or_negative': (self.df['price'] <= 0).sum(),
            'usd_zero_or_negative': (self.df['usdprice'] <= 0).sum(),
        }

        #geographic quality
        nigeria_bounds = {'lat_min': 4, 'lat_max': 14, 'lon_min': 3, 'lon_max': 15}
        valid_coords = (
            (self.df['latitude'].between(nigeria_bounds['lat_min'], nigeria_bounds['lat_max'])) &
            (self.df['longitude'].between(nigeria_bounds['lon_max'], nigeria_bounds['lon_max']))
        )

        quality_metrics['geographic'] = {
            'valid_coordinates': valid_coords.sum(),
            'invalid_coordinates': (~valid_coords).sum()
        }

        #temporal quality
        quality_metrics['temporal'] = {
            'date_range_years': (self.df['date'].max() - self.df['date'].min()).days / 365.25,
            'records_per_year': len(self.df) / ((self.df['date'].max() - self.df['date'].min()).days / 365.25)
        }

        return quality_metrics


    
    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'data_range': (self.df['date'].min(), self.df['date'].max()),
            'unique_markets': self.df['market'].nunique(),
            'unique_commodities': self.df['commodity'].nunique(),
            'unique_admin1': self.df['admin1'].nunique(),
            'missing_value': self.df.isnull().sum().to_dict()
        }