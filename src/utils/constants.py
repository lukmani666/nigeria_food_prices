import plotly.express as px
"""Constants and configuration for Nigeria Food Price analysis"""

#Nigeria geographic bounds
NIGERIA_BOUNDS = {
    'lat_min': 4.0,
    'lat_max': 14.0,
    'lon_min': 3.0,
    'lon_max': 15.0
}

#Major Nigeria states
MAJOR_STATES = [
    'Lagos', 'Kano', 'Kaduna', 'Oyo', 'Rivers', 'Bayelsa', 'Kogi', 'Imo',
    'Anambra', 'Plateau', 'Borno', 'Osun', 'Delta', 'Edo', 'Adamawa'
]

# Common food categories
FOOD_CATEGORIES = [
    'Cereals', 'Tubers', 'Legumes', 'Vegetables', 'Fruits', 'Meat', 'Fish', 'Dairy', 'Oils', 'Spices', 'Beverages'
]

#common commodities
MAJOR_COMMODITIES = [
    'Rice', 'Maize', 'Yam', 'Cassava', 'Beans', 'Tomato', 'Onion', 'Pepper', 'Beef', 'Chicken', 'Fish', 'Groundnut oil', 'Palm oil'
]

#unit standardization mapping
UNIT_MAPPING = {
    # Weight units
    'kg': 'kg',
    'kilogram': 'kg',
    'kilograms': 'kg',
    'kilo': 'kg',
    'gram': 'g',
    'grams': 'g',
    'g': 'g',
    
    # Volume units
    'litre': 'l',
    'liter': 'l',
    'litres': 'l',
    'liters': 'l',
    'l': 'l',
    
    # Count units
    'piece': 'piece',
    'pieces': 'piece',
    'each': 'piece',
    'item': 'piece',
    
    # Bulk units
    'bag': 'bag',
    'bags': 'bag',
    'sack': 'bag',
    'basket': 'basket',
    'bowl': 'bowl'
}

# Color schemes for visualization
COLOR_SCHEMES = {
    'states': px.colors.qualitative.Set3,
    'commodities': px.colors.qualitative.Dark24,
    'categories': px.colors.qualitative.Pastel,
    'sequential': px.colors.sequential.Viridis,
    'diverging': px.colors.diverging.RdBu
}

# Analysis parameters
ANALYSIS_PARAMS = {
    'outlier_threshold': {
        'iqr_multiplier': 3,
        'zscore_threshold': 4,
        'percentile_bounds': (0.01, 0.99)
    },
    'seasonality': {
        'min_periods': 24, # Minimum months for seasonal analysis
        'seasonal_window': 12 # Months for seasonal decomposition
    },
    'volatility': {
        'window': 12, # Months for rolling volatility
        'min_observations': 6
    }
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'default_commodity': 'Rice',
    'default_state': 'Lagos',
    'default_date_range': 12, #months
    'chart_height': 400,
    'map_height': 600
}