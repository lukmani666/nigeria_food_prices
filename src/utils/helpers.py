import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objs as go

def calculate_price_volatility(df: pd.DataFrame,
                               commodity: str,
                               market: Optional[str] = None,
                               window: int = 12) -> pd.Series:
    """Calculate price volatility for a commodity"""
    if market:
        data = df[(df['commodity'] == commodity) & (df['market'] == market)] 
    else:
        data = df[df['commodity'] == commodity]
    
    monthly_prices = data.groupby([data['date'].dt.year, data['date'].dt.month])['price'].mean()
    returns = monthly_prices.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(12)

    return volatility

def detect_seasonal_patterns(df: pd.DataFrame, commodity: str) -> Dict[str, Any]:
    """Detect seasonal patterns in commodity prices"""
    commodity_data = df[df['commodity'] == commodity].copy()
    commodity_data['month'] = commodity_data['date'].dt.month

    monthly_avg = commodity_data.groupby('month')['price'].mean()
    overall_avg = commodity_data['price'].mean()
    seasonal_index = monthly_avg / overall_avg

    return {
        'monthly_average': monthly_avg.to_dict(),
        'seasonal_indices': seasonal_index.to_dict(),
        'peak_months': seasonal_index.nlargest(3).index.tolist(),
        'low_months': seasonal_index.nsmallest(3).index.tolist()
    }

def calulate_market_integration(df: pd.DataFrame,
                                commodity: str,
                                markets: List[str]) -> pd.DataFrame:
    """Calculate market integration metrics between markets"""
    market_data = {}

    for market in markets:
        market_prices = df[(df['commodity'] == commodity) & (df['market'] == market)]
        if not market_prices.empty:
            monthly_avg = market_prices.groupby(['year', 'month'])['price'].mean()
            market_data[market] = monthly_avg
    
    #calculate correlation matrix
    if len(market_data) >= 2:
        price_df = pd.DataFrame(market_data)
        correlation_matrix = price_df.corr()
        return correlation_matrix
    else:
        return pd.DataFrame()
    
def format_currency(value: float, currency: str = 'NGN') -> str:
    """Format currency values for display"""
    if currency == 'NGN':
        return f"₦{value:,.2f}"
    elif currency == 'USD':
        return f"${value:.2f}"
    else:
        return f"{value:,.2f}"

def get_price_percentile(df: pd.DataFrame, commodity: str) -> Dict[str, float]:
    """Get price percentiles for a commodity"""
    commodity_prices = df[df['commodity'] == commodity]['price']

    return {
        'p10': commodity_prices.quantile(0.10),
        'p25': commodity_prices.quantile(0.25),
        'p50': commodity_prices.quantile(0.50),
        'p75': commodity_prices.quantile(0.75),
        'p90': commodity_prices.quantile(0.90),
        'p95': commodity_prices.quantile(0.95),
        'p99': commodity_prices.quantile(0.99)
    }