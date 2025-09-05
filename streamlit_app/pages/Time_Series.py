import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Base project directory (nigeria_food_prices/)
BASE_DIR = Path(__file__).parent.parent.parent

# Add src directory (for your reusable modules)
sys.path.append(str(BASE_DIR / "src"))

try:
    from models.time_series_models import TimeSeriesAnalyzer,PriceForecastModel
    from utils.helpers import detect_seasonal_patterns, calculate_price_volatility
    from utils.constants import MAJOR_COMMODITIES, COLOR_SCHEMES
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError as e:
    st.error(f'Import error: {e}')

st.set_page_config(
    page_title='Time Series Analysis',
    page_icon='📈',
    layout='wide'
)

def prepare_time_series_data(df, commodity, market=None, state=None, aggregation='monthly'):
    """Prepare time series data for analysis"""
    #filter by commodity
    ts_data = df[df['commodity'] == commodity].copy()

    #apply additional filters
    if market:
        ts_data = ts_data[ts_data['market'] == market]
    elif state:
        ts_data = ts_data[ts_data['admin1'] == state]
    
    if ts_data.empty:
        return None
    
    #aggregation data
    if aggregation == 'monthly':
        ts_data = ts_data.groupby([ts_data['date'].dt.year, ts_data['date'].dt.month]).agg({
            'price': 'mean',
            'usdprice': 'mean',
            'date': 'first'
        }).reset_index(drop=True)
    elif aggregation == 'weekly':
        ts_data['week'] = ts_data['date'].dt.isocalendar().week
        ts_data['year'] = ts_data['date'].dt.year
        ts_data = ts_data.groupby(['year', 'week']).agg({
            'price': 'mean',
            'usdprice': 'mean', 
            'date': 'first'
        }).reset_index(drop=True)
    elif aggregation == 'quarterly':
        ts_data = ts_data.groupby([ts_data['date'].dt.year, ts_data['date'].dt.quarter]).agg({
            'price': 'mean',
            'usdprice': 'mean',
            'date': 'first'
        }).reset_index(drop=True)
    
    #sort by date
    ts_data = ts_data.sort_values('date').reset_index(drop=True)

    #create time series index
    ts_data.set_index('date', inplace=True)

    return ts_data

def create_seasonal_decomposition_plot(ts_data, price_col='price', title=''):
    """Create seasonal decomposition visualization"""
    try:
        #ensure we have enough data points
        if len(ts_data) < 24: #need at least 2 years for monthly data
            return None 
        
        #perform seasonal decomposition
        decomposition = seasonal_decompose(ts_data[price_col], model='additive', period=12)

        #create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.08,
            shared_xaxes=True
        )

        #original series
        fig.add_trace(
            go.Scatter(x=ts_data.index, y=ts_data[price_col],
                       mode='lines', name='Original', line=dict(color='blue')),
                       row=1, col=1
        )

        #trend
        fig.add_trace(
            go.Scatter(x=ts_data.index, y=decomposition.trend,
                       mode='lines', name='Trend', line=dict(color='red')),
                       row=2, col=1
        )

        #seasonal
        fig.add_trace(
            go.Scatter(x=ts_data.index, y=decomposition.seasonal,
                       mode='lines', name='Seasonal', line=dict(color='green')),
                       row=3, col=1
        )

        #residual
        fig.add_trace(
            go.Scatter(x=ts_data.index, y=decomposition.resid,
                       mode='lines', name='Residual', line=dict(color='purple')),
                       row=4, col=1
        )

        fig.update_layout(
            height=800,
            title_text=f'Seasonal Decomposition - {title}',
            showlegend=False
        )

        return fig, decomposition
    except Exception as e:
        st.error(f'Error in seasonal decomposition: {e}')
        return None, None

def create_volatility_analysis(ts_data, price_col='price', windows=[3, 6, 12]):
    """Create price volatility analysis"""
    volatility_data = ts_data.copy()

    #calculate returns
    volatility_data['returns'] = volatility_data[price_col].pct_change()

    #calculate rolling volatility for different windows
    for window in windows:
        volatility_data[f'volatility_{window}m'] = (
            volatility_data['returns'].rolling(window=window).std() * np.sqrt(12)
        )

    #create visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Price and Moving Averages', 'Rolling Volatility'],
        vertical_spacing=0.15
    )

    #price with moving averages
    fig.add_trace(
        go.Scatter(x=volatility_data.index, y=volatility_data[price_col],
                   mode='lines', name='Price', line=dict(color='blue')),
                   row=1, col=1
    )

    #add moving averages
    for window in [3, 6, 12]:
        ma = volatility_data[price_col].rolling(window=window).mean()
        fig.add_trace(
            go.Scatter(x=volatility_data.index, y=ma,
                       mode='lines', name=f'{window}M MA',
                       line=dict(dash='dash')),
                       row=1, col=1
        )
    
    #volatility plots
    colors = ['red', 'orange', 'green']
    for i, window in enumerate(windows):
        fig.add_trace(
            go.Scatter(x=volatility_data.index,
                       y=volatility_data[f'volatility_{window}m'],
                       mode='lines', name=f'{window}M Volatility',
                       line=dict(color=colors[i])),
            row=2, col=1
        )
    
    fig.update_layout(height=600, title_text='Price Volatility Analysis')
    return fig, volatility_data

def check_stationarity(ts_data, price_col='price'):
    """Check time serie stationarity using Augmented Dickey-Fuller test"""
    try:
        #remove nan values
        clean_data = ts_data[price_col].dropna()

        if len(clean_data) < 10:
            return None
        
        #perform ADF test
        adf_result = adfuller(clean_data, autolag='AIC')

        return {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] <= 0.05
        }
    except Exception as e:
        st.error(f'Error in stationary test: {e}')
        return None

def create_autocorrelation_plots(ts_data, price_col='price', lags=24):
    """Create ACF and PACF plots"""
    try:
        from statsmodels.tsa.stattools import acf, pacf

        clean_data = ts_data[price_col].dropna()

        if len(clean_data) < lags + 10:
            return None
        
        #calculate ACF and PACF
        acf_values = acf(clean_data, nlags=lags)
        pacf_values = pacf(clean_data, nlags=lags)

        #create plots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Autocorrelation Function (ACF)', 'Partial Autocorrelation Function (PACF)']
        )

        #ACF plot
        fig.add_trace(
            go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF'),
            row=1, col=1
        )

        #PACF plot 
        fig.add_trace(
            go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF'),
            row=2, col=1
        )

        #add confidence intervals
        confidence_interval = 1.96 / np.sqrt(len(clean_data))

        for row in [1, 2]:
            fig.add_hline(y=confidence_interval, line_dash='dash', line_color='red',
                          row=row, col=1)
            fig.add_hline(y=-confidence_interval, line_dash='dash', line_color='red', row=row, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        return fig
    except Exception as e:
        st.error(f'Error creating autocorrelation plots: {e}')
        return None

def main():
    st.title("📈 Time Series Analysis")
    st.markdown("Advance temporal analysis and forecasting for Nigeria food price")

    #check for data
    if not hasattr(st.session_state, 'cleaned_data'):
        st.warning("No cleaned data found. Please return to the main page and clean the dataset first.")
        return
    
    df = st.session_state.cleaned_data

    #sidebar controls
    with st.sidebar:
        selected_state = None
        selected_market = None
        st.markdown('### 🎯 Time Series Controls')

        #commodity selection
        selected_commodity = st.selectbox(
            "Select Commodity",
            options=sorted(df['commodity'].unique()),
            index=0
        )
        #geographic scope
        geo_scope = st.radio(
            'Geographic Scope',
            ['National Average', 'State Level', 'Market Level']
        )

        if geo_scope == 'State Level':
            commodity_states = df[df['commodity'] == selected_commodity]['admin1'].unique()
            selected_state = st.selectbox(
                'Select State',
                options=sorted(commodity_states)
            )
            # selected_state = None
        elif geo_scope == 'Market Level':
            commodity_markets = df[df['commodity'] == selected_commodity]['market'].unique()
            selected_market = st.selectbox(
                'Select Market',
                options=sorted(commodity_markets)
            )
            # selected_state = None
        # else:
        #     selected_state = None
        #     selected_market = None
        
        #aggregation level
        aggregation = st.selectbox(
            'Time Aggregation',
            ['monthly', 'quarterly', 'weekly'],
            index=0
        )

        #analysis type
        analysis_type = st.selectbox(
            'Analysis Type',
            [
                'Price Trends & Patterns',
                'Seasonal Decomposition',
                'Volatility Analysis',
                'Stationarity Testing',
                'Autocorrelation Analysis',
                'Price Forecasting'
            ]
        )

        #date range
        date_range = st.date_input(
            'Date Range',
            value=(df['date'].min(), df['date'].max()),
            min_value=df['date'].min(),
            max_value=df['date'].max()
        )
    
    #prepare time series data
    with st.spinner('Preparing time series data...'):
        #apply data filter
        if len(date_range) == 2:
            filtered_df = df[
                (df['date'] >= pd.to_datetime(date_range[0])) &
                (df['date'] <= pd.to_datetime(date_range[1]))
            ]
        else:
            filtered_df = df
        
        #prepare time series
        ts_data = prepare_time_series_data(
            filtered_df,
            selected_commodity,
            market=selected_market,
            state=selected_state,
            aggregation=aggregation
        )

    if ts_data is None or ts_data.empty:
        st.error('No data available for thr selected filters. Please adjust your selection.')
        return
    
    #display basic time series info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric('Date Points', f'{len(ts_data)}')
    with col2:
        st.metric('Time Span', f"{(ts_data.index.max() - ts_data.index.min()).days} days")
    with col3:
        avg_price = ts_data['price'].mean()
        st.metric("Average Price", f"₦{avg_price:,.2f}")
    with col4:
        price_change = ((ts_data['price'].iloc[-1] - ts_data['price'].iloc[0]) / ts_data['price'].iloc[0]) * 100
        st.metric('Total Change', f'{price_change:+.1f}%')
    
    #analysis based on selected type
    if analysis_type == 'Price Trends & Patterns':
        st.markdown(f"### 📊 Price Trends - {selected_commodity}")

        #main price trend chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=ts_data.index,
                y=ts_data['price'],
                mode='lines+markers',
                name='NGN Price',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            )
        )
        #add moving averages
        ma_windows = [3, 6, 12] if aggregation == 'monthly' else [4, 8, 16]
        colors = ['red', 'orange', 'green']

        for i, window in enumerate(ma_windows):
            if len(ts_data) >= window:
                ma = ts_data['price'].rolling(window=window).mean()
                fig.add_trace(
                    go.Scatter(
                        x=ts_data.index,
                        y=ma,
                        mode='lines',
                        name=f"{window}-Period MA",
                        line=dict(color=colors[i], dash='dash')
                    )
                )
        
        fig.update_layout(
            title=f"{selected_commodity} - Price Trends with Moving Averages",
            xaxis_title='Date',
            yaxis_title='Price (NGN)',
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        #statistical summary
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Price Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                'NGN': [
                    f"₦{ts_data['price'].mean():,.2f}",
                    f"₦{ts_data['price'].median():,.2f}",
                    f"₦{ts_data['price'].std():,.2f}",
                    f"₦{ts_data['price'].min():,.2f}",
                    f"₦{ts_data['price'].max():,.2f}",
                    f"₦{ts_data['price'].max() - ts_data['price'].min():,.2f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📈 Growth Analysis")

            #calculate year-over-year growth if we have enough data
            if len(ts_data) >= 12:
                ts_data_copy = ts_data.copy()
                ts_data_copy['yoy_growth'] = ts_data_copy['price'].pct_change(12) * 100

                recent_growth = ts_data_copy['yoy_growth'].dropna().iloc[-1] if not ts_data_copy['yoy_growth'].dropna().empty else 0
                avg_growth = ts_data_copy['yoy_growth'].mean() if not ts_data_copy['yoy_growth'].isna().all() else 0

                growth_df = pd.DataFrame({
                    'Period': ['Recent YoY', 'Average YoY', 'Total Period'],
                    'Growth Rate': [
                        f'{recent_growth:+.1f}%',
                        f"{avg_growth:+.1f}%",
                        f"{price_change:+.1f}%"
                    ]
                })
                st.dataframe(growth_df, use_container_width=True, hide_index=True)
    
    elif analysis_type == "Seasonal Decomposition":
        st.markdown(f"### 🌊 Seasonal Decomposition - {selected_commodity}")

        if len(ts_data) < 24:
            st.warning("Seasonal decomposition requires at least 2 years of data. Consider changing your filters or aggregation level.")
        else:
            fig, decomposition = create_seasonal_decomposition_plot(
                ts_data,
                title=f"{selected_commodity} {'- ' + selected_state if selected_state else ''}"
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                #seasonal insights
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🔍 Seasonal Insights")

                    #peak and low seasons
                    seasonal_avg = decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean()
                    peak_months = seasonal_avg.nlargest(3).index.tolist()
                    low_months = seasonal_avg.nsmallest(3).index.tolist()

                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                    st.write("**Peak Price Months:**")
                    for month in peak_months:
                        st.write(f"• {month_names[month-1]}")

                    st.write("**Low Price Months:**")
                    for month in low_months:
                        st.write(f"• {month_names[month-1]}")
                
                with col2:
                    st.markdown("#### 📊 Component Statistics")

                    treand_change = ((decomposition.trend.dropna().iloc[-1] - decomposition.trend.dropna().iloc[0]) / 
                                     decomposition.trend.dropna().iloc[0] * 100)
                    seasonal_amplitude = decomposition.seasonal.max() - decomposition.seasonal.min()
                    residual_std = decomposition.resid.std()

                    component_stats = pd.DataFrame({
                        'Component': ['Trend Change', 'Seasonal Amplitude', 'Residual Std Dev'],
                        'Value': [
                            f"{treand_change:+.1f}%",
                            f"₦{seasonal_amplitude:,.2f}",
                            f"₦{residual_std:,.2f}"
                        ]
                    })
                    st.dataframe(component_stats, use_container_width=True, hide_index=True)
    
    elif analysis_type == "Volatility Analysis":
        st.markdown(f"### 📉 Volatility Analysis - {selected_commodity}")
        fig, volatility_data = create_volatility_analysis(ts_data)

        if fig:
            st.plotly_chart(fig, use_container_width=True)

            #volatility statistics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Volatility Metrics")

                current_vol_3m = volatility_data['volatility_3m'].dropna().iloc[-1] if not volatility_data['volatility_3m'].dropna().empty else 0
                current_vol_12m = volatility_data['volatility_12m'].dropna().iloc[-1] if not volatility_data['volatility_12m'].dropna().empty else 0
                avg_vol_12m = volatility_data['volatility_12m'].mean()

                vol_metrics = pd.DataFrame({
                    'Period': ['3-Month (Current)', '12-Month (Current)', '12-Month (Average)'],
                    'Volatility': [
                        f"{current_vol_3m:.1f}%",
                        f"{current_vol_12m:.1f}%", 
                        f"{avg_vol_12m:.1f}%"
                    ]
                })
                st.dataframe(vol_metrics, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### 🎯 Risk Assessment")

                #risk categories based on volatility
                if avg_vol_12m < 0.1:
                    risk_level = "Low",
                    risk_color = "🟢"
                elif avg_vol_12m < 0.3:
                    risk_level = "Moderate"
                    risk_color = "🟡"
                else:
                    risk_level = "High"
                    risk_color = "🔴"
                
                st.write(f"**Risk Level:** {risk_color} {risk_level}")
                st.write(f"**Price Stability:** {'Stable' if avg_vol_12m < 0.2 else 'Volatile'}")

                #recent trend
                recent_volatility = volatility_data['volatility_3m'].dropna().iloc[-3:].mean()
                historical_volatility = volatility_data['volatility_3m'].dropna().iloc[:-3].mean()

                if recent_volatility > historical_volatility * 1.2:
                    trend = "📈 Increasing"
                elif recent_volatility < historical_volatility * 0.8:
                    trend = "📉 Decreasing"
                else:
                    trend = "➡️ Stable"
                st.write(f"**Recent Trend:** {trend}")
    
    elif analysis_type == "Stationarity Testing":
        st.markdown(f"### 🔬 Stationarity Testing - {selected_commodity}")

        stationarity_result = check_stationarity(ts_data)

        if stationarity_result:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Augmented Dickey-Fuller Test")

                #test results
                st.write(f"**ADF Statistic:** {stationarity_result['adf_statistic']:.4f}")
                st.write(f"**P-value:** {stationarity_result['p_value']:.4f}")

                if stationarity_result['is_stationary']:
                    st.success("✅ Time series is stationary")
                else:
                    st.warning("⚠️ Time series is non-stationary")
                
                #critical values
                st.write("**Critical Values:**")
                for level, value in stationarity_result['critical_values'].items():
                    st.write(f"• {level}: {value:.4f}")
            
            with col2:
                st.markdown("### 💡 Interpretation")

                if stationarity_result['is_stationary']:
                    st.write("The time series is **stationary**, meaning:")
                    st.write("• Mean and variance are constant over time")
                    st.write("• Good for forecasting models")
                    st.write("• No need for differencing")
                else:
                    st.write("The time series is **non-stationary**, meaning:")
                    st.write("• Trending or seasonal patterns present")
                    st.write("• May need differencing for modeling")
                    st.write("• Consider seasonal decomposition")
        
        #show differenced series if non-stationary
        if stationarity_result and not stationarity_result['is_stationary']:
            st.markdown("#### 🔄 Differenced Series")

            # First difference
            ts_diff = ts_data['price'].diff(). dropna()

            #test stationarity of differenced series
            diff_stationarity = check_stationarity(pd.DataFrame({'price': ts_diff}))

            col1, col2 = st.columns(2)

            with col1:
                #plot original vs differenced
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['Original Series', 'First Differenced Series']
                )

                fig.add_trace(
                    go.Scatter(x=ts_data.index, y=ts_data['price'],
                               mode='lines', name='Original'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=ts_diff.index, y=ts_diff.values, 
                              mode='lines', name='Differenced'),
                    row=2, col=1
                )
                
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if diff_stationarity:
                    st.write("**First Difference Results:**")
                    st.write(f"ADF Statistic: {diff_stationarity['adf_statistic']:.4f}")
                    st.write(f"P-value: {diff_stationarity['p_value']:.4f}")
                    
                    if diff_stationarity['is_stationary']:
                        st.success("✅ Differenced series is stationary")
                        st.write("Ready for ARIMA modeling with d=1")
                    else:
                        st.warning("⚠️ May need seasonal differencing")
    
    elif analysis_type == "Autocorrelation Analysis":
        st.markdown(f"### 🔗 Autocorrelation Analysis - {selected_commodity}")

        fig = create_autocorrelation_plots(ts_data)

        if fig:
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 💡 ACF Interpretation")
                st.write("The **Autocorrelation Function (ACF)** shows:")
                st.write("• Correlation between price and its lagged values")
                st.write("• Slow decay suggests non-stationarity")
                st.write("• Significant spikes indicate seasonal patterns")
                st.write("• Helps determine MA(q) order in ARIMA")
            
            with col2:
                st.markdown("#### 🎯 PACF Interpretation")
                st.write("The **Partial ACF (PACF)** shows:")
                st.write("• Direct correlation after removing intermediate lags")
                st.write("• Sharp cutoff suggests AR process")
                st.write("• Helps determine AR(p) order in ARIMA")
                st.write("• Useful for model specification")
        else:
            st.error("Unable to create autocorrelation plots. Need more data points.")
    
    elif analysis_type == "Price Forecasting":
        st.markdown(f"### 🔮 Price Forecasting - {selected_commodity}")

        if len(ts_data) < 24:
            st.warning("Forecasting requires at least 2 years of data for reliable results.")
        else:
            # Forecasting controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                forecast_periods = st.slider("Forecast Periods", 1, 12, 6)
            with col2:
                model_type = st.selectbox("Model Type", ["Auto ARIMA", "Linear Trend"])
            with col3:
                confidence_level = st.slider("Confidence Level", 80, 99, 95)
            
            if st.button("Generate Forecast", type='primary'):
                with st.spinner("Training model and generating forecast..."):
                    try:
                        #simple forecast implementation
                        from sklearn.linear_model import LinearRegression

                        #prepare data for modeling
                        ts_reset = ts_data.reset_index()
                        ts_reset['time_idx'] = range(len(ts_reset))

                        #split data (use last 80% for training)
                        split_idx = int(len(ts_reset) * 0.8)
                        train_data = ts_reset.iloc[:split_idx]
                        test_data = ts_reset.iloc[split_idx:]

                        test_pred = np.array([])

                        if model_type == "Linear Trend":
                            #simple linear regression
                            model = LinearRegression()
                            model.fit(train_data[['time_idx']], train_data['price'])

                            #make predictions on the test set
                            test_pred = model.predict(test_data[['time_idx']])

                            #generate future predictions
                            future_idx = range(len(ts_reset), len(ts_reset) + forecast_periods)
                            future_pred = model.predict(np.array(future_idx).reshape(-1, 1))

                            #create future dates
                            last_date = ts_data.index[-1]
                            if aggregation == 'monthly':
                                future_dates = pd.date_range(start=last_date, periods=forecast_periods+1, freq='M')[1:]
                            elif aggregation == 'quarterly':
                                future_dates = pd.date_range(start=last_date, periods=forecast_periods+1, freq='Q')[1:]
                            else:
                                future_dates = pd.date_range(start=last_date, periods=forecast_periods+1, freq='W')[1:]
                            
                            #calculate confidence intervals (simple approach)
                            residuals = train_data['price'] - model.predict(train_data[['time_idx']])
                            residual_std = np.std(residuals)
                            confidence_multiplier = 1.96 if confidence_level == 95 else 2.56

                            upper_bound = future_pred + (confidence_multiplier * residual_std)
                            lower_bound = future_pred - (confidence_multiplier * residual_std)
                            
                        elif model_type == "Auto ARIMA":
                            from statsmodels.tsa.arima.model import ARIMA
                            # arima_model = auto_arima(
                            #     train_data['price'],
                            #     seasonal=False,
                            #     trace=False,
                            #     error_action='ignore',
                            #     suppress_warnings=True
                            # )
                            # Fit ARIMA model (p,d,q) — start with something simple like (1,1,1)
                            model = ARIMA(train_data['price'], order=(1,1,1))
                            model_fit = model.fit()

                            # Forecast
                            forecast_all = model_fit.get_forecast(steps=forecast_periods)
                            forecast_mean = forecast_all.predicted_mean
                            conf_int = forecast_all.conf_int(alpha=1 - confidence_level/100)

                            # Split into test predictions and future predictions
                            if not test_data.empty:
                                test_pred = model_fit.get_forecast(steps=len(test_data)).predicted_mean.values
                            else:
                                test_pred = np.array([])

                            # Extract future dates and forecast
                            future_dates = pd.date_range(
                                start=train_data.index[-1], 
                                periods=forecast_periods+1, 
                                freq='M' if aggregation=='monthly' else 'Q' if aggregation=='quarterly' else 'W'
                            )[1:]
                            future_pred = forecast_mean.values
                            lower_bound = conf_int.iloc[:, 0].values
                            upper_bound = conf_int.iloc[:, 1].values

                            # predict on test set
                            # if not test_data.empty:
                            #     test_pred, conf_int = arima_model.predict(
                            #         n_periods=len(test_data),
                            #         return_conf_int=True
                            #     )
                            # else:
                            #     test_pred, conf_int = [], []

                            # # forecast future
                            # future_pred, conf_int = arima_model.predict(
                            #     n_periods=forecast_periods,
                            #     return_conf_int=True
                            # )

                            # # future dates
                            # last_date = ts_data.index[-1]
                            # if aggregation == 'monthly':
                            #     future_dates = pd.date_range(start=last_date, periods=forecast_periods+1, freq='M')[1:]
                            # elif aggregation == 'quarterly':
                            #     future_dates = pd.date_range(start=last_date, periods=forecast_periods+1, freq='Q')[1:]
                            # else:
                            #     future_dates = pd.date_range(start=last_date, periods=forecast_periods+1, freq='W')[1:]

                            # # confidence intervals from ARIMA
                            # lower_bound = conf_int[:, 0]
                            # upper_bound = conf_int[:, 1]
                        
                        #create forecast visualization
                        fig = go.Figure()

                        #historical data
                        fig.add_trace(
                            go.Scatter(
                                x=train_data['date'],
                                y=train_data['price'],
                                mode='lines',
                                name='Historical (Training)',
                                line=dict(color='blue')
                            )
                        )

                        #test data (if available)
                        if not test_data.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=test_data['date'],
                                    y=test_data['price'],
                                    mode='lines',
                                    name='Historical (Test)',
                                    line=dict(color='green')
                                )
                            )

                            #test predictions
                            fig.add_trace(
                                go.Scatter(
                                    x=test_data['date'],
                                    y=test_pred,
                                    mode='lines',
                                    name='Test Predictions',
                                    line=dict(color='orange', dash='dash')
                                )
                            )
                        
                        #future predictions
                        fig.add_trace(
                            go.Scatter(
                                x=future_dates,
                                y=future_pred,
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='red', width=3)
                            )
                        )

                        #confidence intervals
                        fig.add_trace(
                            go.Scatter(
                                x=future_dates,
                                y=upper_bound,
                                fill=None,
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                showlegend=False
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=future_dates,
                                y=lower_bound,
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                name=f'{confidence_level}% Confidence Interval',
                                fillcolor='rgba(255,0,0,0.2)'
                            )
                        )

                        fig.update_layout(
                            title=f'{selected_commodity} Price Forcast - {model_type}',
                            xaxis_title='Date',
                            yaxis_title='Price (NGN)',
                            height=500,
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        #forecast results table
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### 🔮 Forecast Results")
                            
                            forecast_df = pd.DataFrame({
                                'Date': future_dates,
                                'Forecast': [f"₦{pred:,.2f}" for pred in future_pred],
                                'Lower Bound': [f"₦{lower:,.2f}" for lower in lower_bound],
                                'Upper Bound': [f"₦{upper:,.2f}" for upper in upper_bound]
                            })
                            
                            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            st.markdown("#### 📊 Model Performance")
                            
                            if not test_data.empty:
                                # Calculate metrics
                                mae = mean_absolute_error(test_data['price'], test_pred)
                                rmse = np.sqrt(mean_squared_error(test_data['price'], test_pred))
                                mape = np.mean(np.abs((test_data['price'] - test_pred) / test_data['price'])) * 100
                                
                                metrics_df = pd.DataFrame({
                                    'Metric': ['MAE', 'RMSE', 'MAPE'],
                                    'Value': [
                                        f"₦{mae:,.2f}",
                                        f"₦{rmse:,.2f}",
                                        f"{mape:.2f}%"
                                    ]
                                })
                                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No test data available for performance metrics")
                            
                            # Forecast summary
                            current_price = ts_data['price'].iloc[-1]
                            forecast_change = ((future_pred[-1] - current_price) / current_price) * 100
                            
                            st.write(f"**Current Price:** ₦{current_price:,.2f}")
                            st.write(f"**Final Forecast:** ₦{future_pred[-1]:,.2f}")
                            st.write(f"**Expected Change:** {forecast_change:+.1f}%")
                        
                        # Forecast interpretation
                        st.markdown("#### 💡 Forecast Interpretation")
                        
                        if forecast_change > 10:
                            interpretation = "📈 **Strong Price Increase Expected** - Consider supply chain preparations"
                        elif forecast_change > 5:
                            interpretation = "📊 **Moderate Price Increase** - Monitor market conditions"
                        elif forecast_change < -10:
                            interpretation = "📉 **Significant Price Decrease** - Possible oversupply or demand reduction"
                        elif forecast_change < -5:
                            interpretation = "📊 **Moderate Price Decrease** - Favorable market conditions"
                        else:
                            interpretation = "➡️ **Price Stability Expected** - Normal market conditions"
                        
                        st.write(interpretation)
                    except Exception as e:
                        st.error(f'Forecasting error: {e}')
                        st.info('Try adjusting your data filters or selecting a different model type.')
    
    st.markdown('---')
    st.caption('Built with ❤️ - All rights reserved © Lukman Olamide Ademola')

if __name__ == "__main__":
    main()