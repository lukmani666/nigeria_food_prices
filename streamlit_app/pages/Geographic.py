import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Base project directory (nigeria_food_prices/)
BASE_DIR = Path(__file__).parent.parent

# Add project root (for config.py, etc.)
sys.path.append(str(BASE_DIR))

# Add src directory (for your reusable modules)
sys.path.append(str(BASE_DIR / "src"))

try:
    from utils.helpers import format_currency, calulate_market_integration
    from utils.constants import NIGERIA_BOUNDS, MAJOR_COMMODITIES, COLOR_SCHEMES
except ImportError:
    st.error("Unable to import utility functions. Please check project structure.")

st.set_page_config(
    page_title="Geographic Analysis",
    page_icon="🗺️",
    layout="wide"
)

def create_price_map(df, commodity, price_type='price'):
    """Create an interactive price map"""
    commodity_data = df[df['commodity'] == commodity].copy()

    if commodity_data.empty:
        return None
    
    #aggregate prices by market location
    market_prices = commodity_data.groupby(['market', 'latitude', 'longitude']).agg({
        price_type: 'mean',
        'admin1': 'first',
        'admin2': 'first'
    }).reset_index()

    #remove invalid coordinates
    market_prices = market_prices[
        (market_prices['latitude'].between(NIGERIA_BOUNDS['lat_min'], NIGERIA_BOUNDS['lat_max'])) &
        (market_prices['longitude'].between(NIGERIA_BOUNDS['lon_min'], NIGERIA_BOUNDS['lon_max']))
    ]

    if market_prices.empty:
        return None
    
    #create map
    fig = px.scatter_mapbox(
        market_prices,
        lat='latitude',
        lon='longitude',
        size=price_type,
        color=price_type,
        hover_name='market',
        hover_data={'admin1': True, 'admin2': True, price_type: ':.2f'},
        color_continuous_scale='viridis',
        title=f"{commodity} - Average Prices by Market Location",
        zoom=5,
        height=600
    )

    #update map layout
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=9.5, lon=8.5), #center of Nigeria
        ),
        margin={'r': 0, 't': 50, 'l': 0, 'b': 0}
    )

    return fig

def create_state_comparison_chart(df, commodities, metric='mean'):
    """Create state comparison chart for selected commodities"""
    comparison_data = []

    for commodity in commodities:
        commodity_df = df[df['commodity'] == commodity]

        if metric == 'mean':
            state_stats = commodity_df.groupby('admin1')['price'].mean()
        elif metric == 'median':
            state_stats = commodity_df.groupby('admin1')['price'].median()
        elif metric == 'volatility':
            state_stats = commodity_df.groupby('admin1')['price'].std()
        
        for state, value in state_stats.items():
            comparison_data.append({
                'State': state,
                'Commodity': commodity,
                'Value': value
            })
    
    comparison_df = pd.DataFrame(comparison_data)

    if comparison_df.empty:
        return None
    
    fig = px.bar(
        comparison_df,
        x='State',
        y='Value',
        color='Commodity',
        title=f'{metric.title()} Prices by State and Commodity',
        barmode='group',
        height=500
    )

    fig.update_xaxes(tickangle=45)
    return fig

def calculate_price_disparities(df, commodity):
    """Calculate price disparities across markets"""
    commodity_data = df[df['commodity'] == commodity]

    if commodity_data.empty:
        return None
    
    #calculate market-level statistics
    market_stats = commodity_data.groupby(['admin1', 'market']).agg({
        'price': ['mean', 'std', 'count'],
        'latitude': 'first',
        'longitude': 'first'
    }).round(2)

    market_stats.columns = ['avg_price', 'price_std', 'records', 'latitude', 'longitude']
    market_stats = market_stats.reset_index()

    #calculate coefficient of variation
    market_stats['cv'] = (market_stats['price_std'] / market_stats['avg_price']) * 100

    #state-level aggregation
    state_stats = commodity_data.groupby('admin1').agg({
        'price': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)

    state_stats.columns = ['avg_price', 'price_std', 'min_price', 'max_price', 'records']
    state_stats = state_stats.reset_index()
    state_stats['price_range'] = state_stats['max_price'] - state_stats['min_price']
    state_stats['cv'] = (state_stats['price_std'] / state_stats['avg_price']) * 100

    return market_stats, state_stats


def main():
    st.title("🗺️ Geographic Analysis")
    st.markdown("Explore spatial patterns and regional variations in food price across Nigeria")

    #check for dat
    if not hasattr(st.session_state, 'cleaned_data'):
        st.warning("No cleaned data found. Please return to main page and clean the dataset first.")
        return
    
    df = st.session_state.cleaned_data

    #sidebar controls
    with st.sidebar:
        st.markdown("### 🎯 Analysis Controls")

        #commodity selection
        selected_commodity = st.selectbox(
            "Select Commodity",
            options=sorted(df['commodity'].unique()),
            index=0 if len(df['commodity'].unique()) > 0 else None
        )

        #analysis type
        analysis_type = st.radio(
            "Analysis Type",
            ['Price Distribution Map', 'State Comparisons', 'Market Integration', 'Price Disparities']
        )

        #additional filters
        if analysis_type == 'State Comparisons':
            comparison_commodities = st.multiselect(
                "Select Commodities for Comparison",
                options=sorted(df['commodity'].unique()),
                default=[selected_commodity] if selected_commodity else [],
                max_selections=5
            )

            comparison_metric = st.selectbox(
                'Comparison Metric',
                ["mean", "median", "volatility"]
            )

        #date range filter
        date_range = st.date_input(
            "Date Range",
            value=(df['date'].min(), df['date'].max()),
            min_value=df['date'].min(),
            max_value=df['date'].max()
        )
    
    #apply date filter
    if len(date_range) == 2:
        filtered_df = df[
            (df['date'] >= pd.to_datetime(date_range[0])) &
            (df['date'] <= pd.to_datetime(date_range[1]))
        ]
    else:
        filtered_df = df
    
    #main content based on analysis type
    if analysis_type == "Price Distribution Map":
        st.markdown(f"### 📍 {selected_commodity} - Price Distribution Map")

        #price type selection
        col1, col2 = st.columns([3, 1])
        with col2:
            price_type = st.radio("Price Currency", ["price", "usdprice"], format_func=lambda x: "NGN" if x == "price" else "USD")
        
        #create map
        fig = create_price_map(filtered_df, selected_commodity, price_type)

        if fig:
            st.plotly_chart(fig, use_container_width=True)

            #summary statistics
            commodity_data = filtered_df[filtered_df['commodity'] == selected_commodity]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Markets", f"{commodity_data['market'].nunique()}")
            with col2:
                avg_price = commodity_data[price_type].mean()
                currency = "NGN" if price_type == 'price' else "USD"
                st.metric(f"Average Price ({currency})", format_currency(avg_price, currency))
            with col3:
                price_std = commodity_data[price_type].std()
                st.metric(f"Price Std Dev", format_currency(price_std, currency))
            with col4:
                cv = (price_std / avg_price) * 100 if avg_price > 0 else 0
                st.metric("Coefficient of Variation", f"{cv:.1f}%")
        else:
            st.error("No valid geographic data found for the selected commodity and date range.")
    elif analysis_type == 'State Comparisons':
        if comparison_commodities:
            st.markdown(f"### 📊 State-wise Price Comparison")
            
            #create comparison chart
            fig = create_state_comparison_chart(filtered_df, comparison_commodities, comparison_metric)

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                #detailed comparison table
                st.markdown("### 📋 Detailed State Comparison")

                comparison_data = []
                for commodity in comparison_commodities:
                    commodity_df = filtered_df[filtered_df['commodity'] == commodity]

                    state_stats = commodity_df.groupby('admin1').agg({
                        'price': ['count', 'mean', 'median', 'std', 'min', 'max']
                    }).round(2)

                    state_stats.columns = ['Records', 'Mean', 'Median', 'StdDev', 'Min', 'Max']
                    state_stats['Commodity'] = commodity
                    state_stats['CV%'] = ((state_stats['StdDev'] / state_stats['Mean']) * 100).round(1)
                    state_stats = state_stats.reset_index()
                    comparison_data.append(state_stats)
                
                if comparison_data:
                    combined_stats = pd.concat(comparison_data, ignore_index=True)
                    st.dataframe(combined_stats, use_container_width=True)
            else:
                st.error("No data available for the selected commodities.")
        else:
            st.info("Please select commodities for comparison.")

    elif analysis_type == "Market Integration":
        st.markdown(f"### 🔗 Market Integration Analysis - {selected_commodity}")

        #select markets for integration analysis
        commodity_markets = filtered_df[filtered_df['commodity'] == selected_commodity]['market'].unique()

        if len(commodity_markets) < 2:
            st.error("Need at least 2 markets with data for integration analysis")
            return
        
        selected_markets = st.multiselect(
            "Select Markets for Integration Analysis",
            options=sorted(commodity_markets),
            default=sorted(commodity_markets)[:min(5, len(commodity_markets))],
            max_selections=10
        )

        if len(selected_markets) >= 2:
            #calculate correlation matrix
            correlation_matrix = calulate_market_integration(filtered_df, selected_commodity, selected_markets)

            if not correlation_matrix.empty:
                col1, col2 = st.columns([2, 1])

                with col1:
                    #correlation heatmap
                    fig = px.imshow(
                        correlation_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Market Price Correlation Matrix",
                        color_continuous_scale="RdBu",
                        zmin=-1, zmax=1,
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    #integration statistics
                    st.markdown("### 📊 Integration Metrics")

                    #average correlation
                    avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
                    st.metric("Average Correlation", f"{avg_corr:.3f}")

                    #highest correlation pair
                    corr_copy = correlation_matrix.copy()
                    np.fill_diagonal(corr_copy.values, np.nan)
                    max_corr = corr_copy.max().max()
                    max_pair = corr_copy.stack().idxmax()

                    st.metric("Highest Correlation", f"{max_corr:.3f}")
                    st.write(f"**Markets:** {max_pair[0]} ↔ {max_pair[1]}")

                    #lowest correlation pair
                    min_corr = corr_copy.min().min()
                    min_pair = corr_copy.stack().idxmin()

                    st.metric("Lower Correlation", f"{min_corr:.3f}")
                    st.write(f"**Markets:** {min_pair[0]} ↔ {min_pair[1]}")
            
                #price trends comparison
                st.markdown("### 📈 Price Trends Comparison")

                price_trends = []
                for market in selected_markets:
                    market_data = filtered_df[
                        (filtered_df['commodity'] == selected_commodity) &
                        (filtered_df['market'] == market)
                    ]
                    monthly_avg = market_data.groupby(['year', 'month']).agg({
                        'price': 'mean'
                    }).reset_index()
                    monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))
                    monthly_avg['market'] = market
                    price_trends.append(monthly_avg)
                
                if price_trends:
                    combined_trends = pd.concat(price_trends, ignore_index=True)

                    fig = px.line(
                        combined_trends,
                        x='date',
                        y='price',
                        color='market',
                        title=f"Price Trends - {selected_commodity}",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to calculate market integration. Insufficient data.")
        else:
            st.info("Please select at least 2 markets for integration analysis.")
    elif analysis_type == "Price Disparities":
        st.markdown(f"### 📊 Price Disparities Analysis - {selected_commodity}") 
        #calculated disparities
        disparity_results = calculate_price_disparities(filtered_df, selected_commodity)

        if disparity_results:
            market_stats, state_stats = disparity_results

            #state-level disparities
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🏛️ State-Level Price Disparities")

                #price range chart
                fig = px.bar(
                    state_stats.sort_values('price_range', ascending=False).head(15),
                    x='price_range',
                    y='admin1',
                    orientation='h',
                    title='Price Range by State (NGN)',
                    color='price_range',
                    color_continuous_scale='reds'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Coefficient of Variation")

                #cv chart
                fig = px.bar(
                    state_stats.sort_values('cv', ascending=True).head(15),
                    x='cv',
                    y='admin1',
                    orientation='h',
                    title="Price Volatility by State (%)",
                    color='cv',
                    color_continuous_scale='blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            #detailed statistics table
            st.markdown("### 📋 Detailed State Statistics")

            #add ranking columns
            state_stats['Price_Rank'] = state_stats['avg_price'].rank(ascending=False).astype(int)
            state_stats['Volatility_Rank'] = state_stats['cv'].rank(ascending=True).astype(int)

            #format for display
            display_stats = state_stats.copy()
            display_stats['avg_price'] = display_stats['avg_price'].apply(lambda x: f"₦{x:,.2f}")
            display_stats['min_price'] = display_stats['min_price'].apply(lambda x: f"₦{x:,.2f}")
            display_stats['max_price'] = display_stats['max_price'].apply(lambda x: f"₦{x:,.2f}")
            display_stats['price_range'] = display_stats['price_range'].apply(lambda x: f"₦{x:,.2f}")
            display_stats['cv'] = display_stats['cv'].apply(lambda x: f"{x:.1f}%")

            #rename columns for display
            display_stats = display_stats.rename(columns={
                'admin1': 'State',
                'avg_price': 'Avg_Price',
                'min_price': 'Min_Price',
                'max_price': 'Max_Price',
                'price_range': 'Price_Range',
                'cv': 'Volatility_%',
                'records': 'Records'
            })
            st.dataframe(
                display_stats[[
                    'State', 'Avg_Price', 'Min_Price', 'Max_Price', 'Price_Range',
                    'Volatility_%', 'Records', 'Price_Rank', 'Volatility_Rank'
                ]],
                use_container_width=True
            )

            #market-level analysis
            st.markdown("### 🏪 Market-Level Analysis")

            #top markets by price
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🔝 Highest Priced Markets**")
                top_markets = market_stats.nlargest(10, 'avg_price')[['admin1', 'market', 'avg_price', 'records']]
                top_markets['avg_price'] = top_markets['avg_price'].apply(lambda x: f"₦{x:,.2f}")
                st.dataframe(top_markets, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**💰 Lowest Priced Markets**")
                low_markets = market_stats.nsmallest(10, 'avg_price')[['admin1', 'market', 'avg_price', 'records']]
                low_markets['avg_price'] = low_markets['avg_price'].apply(lambda x: f"₦{x:,.2f}")
                st.dataframe(low_markets, use_container_width=True, hide_index=True)
            
            #summary insights
            st.markdown("### 💡 Key Insights")

            highest_state = state_stats.loc[state_stats['avg_price'].idxmax(), 'admin1']
            lowest_state = state_stats.loc[state_stats['avg_price'].idxmin(), 'admin1']
            most_volatile = state_stats.loc[state_stats['cv'].idxmax(), 'admin1']
            least_volatile = state_stats.loc[state_stats['cv'].idxmin(), 'admin1']

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.info(f"**Highest Prices**\n{highest_state}")
            with col2:
                st.success(f"**Lowest Price**\n{lowest_state}")
            with col3:
                st.warning(f"**Most Volatile**\n{most_volatile}")
            with col4:
                st.info(f"**Least Volatile**\n{least_volatile}")
        else:
            st.error('Unable to calculate price disparities. No data available for the selected commodity.')
    
    st.markdown('---')
    st.caption('Built with ❤️ - All rights reserved © Lukman Olamide Ademola')

if __name__ == "__main__":
    main()