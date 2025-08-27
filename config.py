import os
from pathlib import Path

#project path
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODEL_DATA_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

#data file paths
MAIN_DATASET = RAW_DATA_DIR / "nigerian_food_prices_2002_2025.csv"

#model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

#streamlit configuration
PAGE_TITLE = "Nigeria Food Prices Analysis"
PAGE_ICON = "🍽️"
LAYOUT = "wide"

#plotly_theme
PLOTLY_THEME = "ploty_white"

#currency and units
BASE_CURRENCY = "NGN"
USD_CURRENCY = "USD"

#geographic bounds for Nigeria
NIGERIA_BOUNDS = {
    "lat_min": 4.0,
    "lat_max": 14.0,
    "lon_min": 3.0,
    "lon_max": 15.0
}
