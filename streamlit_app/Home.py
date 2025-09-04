import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Base project directory (nigeria_food_prices/)
BASE_DIR = Path(__file__).parent.parent

# Add project root (for config.py, etc.)
sys.path.append(str(BASE_DIR))

# Add src directory (for your reusable modules)
sys.path.append(str(BASE_DIR / "src"))

try:
    from config import PAGE_TITLE, PAGE_ICON, LAYOUT
    from data.data_loader import DataLoader
    from data.data_cleaner import DataCleaner
    from utils.helpers import format_currency, detect_seasonal_patterns
    from utils.constants import MAJOR_STATES, MAJOR_COMMODITIES, DASHBOARD_CONFIG
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure the project structure is set up correctly and all modules are available.")


# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

#custom css
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .sidebar-info {
        background-color: #0E1117;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try: 
        data_path = BASE_DIR / "data" / "raw" / "nigerian_food_prices_2002_2025.csv"
        loader = DataLoader(data_path)
        df = loader.load_data()
        metadata = loader.get_metadata()
        return df, metadata, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def clean_data(df):
    """Clean and preprocess the data"""
    try:
        cleaner = DataCleaner(df)
        cleaned_df = cleaner.clean_all()
        cleaning_report = cleaner.get_cleaning_report()
        return cleaned_df, cleaning_report, None
    except Exception as e:
        return None, None, str(e)

def main():
    # st.title("🍽️ Nigerian Food Prices Analysis Dashboard")
    # Header
    st.markdown('<h1 class="main-header">🍽️ Nigerian Food Prices Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("""
    ### Welcome to the Nigerian Food Prices Analysis Platform
    
    This comprehensive dashboard provides insights into food price trends across Nigeria from 2002 to 2025.
    
    **Key Features:**
    - 📊 **Overview**: Summary statistics and key metrics
    - 🗺️ **Geographic Analysis**: Spatial price variations and market mapping
    - 📈 **Time Series**: Temporal trends and seasonal patterns
    - 🤖 **ML Models**: Predictive models and forecasting
    - 💡 **Insights**: Key findings and recommendations
    
    Use the sidebar to navigate between different analysis sections.
    """)

    with st.sidebar:
        st.markdown("### 🔧 Data Controls")

        #load data
        with st.spinner("Loading dataset..."):
            raw_data, metadata, load_error = load_data()

        if load_error:
            st.error(f"Error loading data: {load_error}")
            st.info("Please ensure your dataset is placed at: `data/raw/nigerian_food_prices_2002_2025.csv`")
            return
            
        if raw_data is not None:
            st.success(f"✅ Dataset loaded: {raw_data.shape[0]:,} records")

            #data cleaning options
            st.markdown("### 🧹 Data Cleaning Options")

            col1, col2 = st.columns(2)
            with col1:
                remove_duplicates = st.checkbox("Remove duplicates", value=True)
                handle_outliers = st.selectbox(
                    "Handle outliers",
                    ["keep", "clip", "drop"],
                    index=1
                )
            
            with col2:
                handle_missing = st.selectbox(
                    "Missing prices",
                    ["drop", "interpolate", "keep"],
                    index=0
                )
                handle_coords = st.selectbox(
                    "Invalid coordinates",
                    ["drop", "impute", "keep"],
                    index=0
                )
            
            #clean data button
            if st.button("🚀 Clean Data", type='primary'):
                with st.spinner("Cleaning data..."):
                    cleaned_data, cleaning_report, clean_error = clean_data(raw_data)

                    if clean_error:
                        st.error(f"Cleaning error: {clean_error}")
                        return
                
                #store cleaned data in session state
                st.session_state.cleaned_data = cleaned_data
                st.session_state.cleaning_report = cleaning_report
                st.success("✅ Data cleaned successfully!")
        
        #display data info
        if hasattr(st.session_state, 'cleaned_data'):
            df = st.session_state.cleaned_data
            st.markdown("### 📊 Dataset Overview")
            st.markdown(f"""
            <div class="sidebar-info">
            <strong>Records:</strong> {len(df):,}<br>
            <strong>Date Range:</strong> {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}<br>
            <strong>States:</strong> {df['admin1'].nunique()}<br>
            <strong>Markets:</strong> {df['market'].nunique()}<br>
            <strong>Commodities:</strong> {df['commodity'].nunique()}
            </div>
            """, unsafe_allow_html=True)
        elif raw_data is not None:
            st.markdown("### 📊 Raw Dataset Info")
            st.markdown(f"""
            <div class="sidebar-info">
            <strong>Records:</strong> {len(raw_data):,}<br>
            <strong>Columns:</strong> {len(raw_data.columns)}<br>
            <strong>Missing Values:</strong> {raw_data.isnull().sum().sum():,}
            </div>
            """, unsafe_allow_html=True)
    
    #main content area
    if not hasattr(st.session_state, 'cleaned_data') and raw_data is not None:
        #show raw data overview
        st.markdown("## 📋 Raw Dataset Overview")
        st.info("👈 Use the sidebar controls to clean the data and unlock full analysis capabilities!")

        #show basic stats
        #quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", f"{len(raw_data):,}")
        with col2:
            st.metric("📅 Year Covered", f"{raw_data['date'].dt.year.nunique()}")
        with col3:
            st.metric("🌾 Commodities", f"{raw_data['commodity'].nunique()}")
        with col4:
            st.metric("🏪 Markets", f"{raw_data['market'].nunique()}")
        
        #quick preview
        st.markdown("### 🔍 Data Preview")
        st.dataframe(raw_data.head(10), use_container_width=True)

        #basic visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Records Over Time")
            yearly_counts = raw_data['date'].dt.year.value_counts().sort_values()
            fig = px.line(
                x=yearly_counts.index,
                y=yearly_counts.values,
                title="Number of Records per Year"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🗺️ Top 10 States by Records")
            top_states = raw_data['admin1'].value_counts().head(10)
            fig = px.bar(
                x=top_states.values,
                y=top_states.index,
                orientation='h',
                title="Records by State"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    elif hasattr(st.session_state, 'cleaned_data'):
        #show cleaned data analysis
        df = st.session_state.cleaned_data
        cleaning_report = st.session_state.cleaning_report

        #key metrics
        st.markdown("## 📊 Key Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            retention_rate = cleaning_report['retention_rate']
            st.metric(
                "🧹 Data Retention",
                f"{retention_rate:.1f}%",
                delta=f"{retention_rate - 100:.1f}%"
            )
        
        with col2:
            avg_price_ngn = df['price'].mean()
            st.metric(
                "💰 Avg Price (NGN)",
                format_currency(avg_price_ngn, 'NGN')
            )
        
        with col3:
            avg_price_usd = df['usdprice'].mean()
            st.metric(
                "💵 Avg Price (USD)",
                format_currency(avg_price_usd, 'USD')
            )
        
        with col4:
            date_span = (df['date'].max() - df['date'].min()).days / 365.25
            st.metric(
                "📅 Time Span",
                f"{date_span:.1f} years"
            )
        
        with col5:
            records_per_month = len(df) / (date_span * 12)
            st.metric(
                "📈 Records/Month",
                f"{records_per_month:.0f}"
            )
        
        #interactive filters
        st.markdown("## 🔍 Interactive Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_commodities = st.multiselect(
                "Select Commodities",
                options=sorted(df['commodity'].unique()),
                default=[df['commodity'].value_counts().index[0]],
                key="commodity_filter"
            )
        
        with col2:
            selected_states = st.multiselect(
                "Select States",
                options=sorted(df['admin1'].unique()),
                default=sorted(df['admin1'].unique())[:5],
                key='state_filter'
            )
        
        with col3:
            date_range = st.date_input(
                "Date Range",
                value=(df['date'].min(), df['date'].max()),
                min_value=df['date'].min(),
                max_value=df['date'].max(),
                key='date_filter'
            )
        
        if selected_commodities and selected_states:
            filtered_df = df[
                (df['commodity'].isin(selected_commodities)) &
                (df['admin1'].isin(selected_states))
            ]

            if len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df['date'] >= pd.to_datetime(date_range[0])) &
                    (filtered_df['date'] <= pd.to_datetime(date_range[1]))
                ]
            
            if not filtered_df.empty:
                #price trends
                st.markdown("### 📈 Price Trends")

                #monthly average prices
                monthly_prices = filtered_df.groupby(['year', 'month', 'commodity'])['price'].mean().reset_index()
                monthly_prices['date'] = pd.to_datetime(monthly_prices[['year', 'month']].assign(day=1))

                fig = px.line(
                    monthly_prices,
                    x='date',
                    y='price',
                    color='commodity',
                    title='Average Monthly Prices (NGN)',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                #geographic distraction
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🗺️ Price by State")
                    state_price = filtered_df.groupby('admin1')['price'].mean().sort_values(ascending=True)

                    fig = px.bar(
                        x=state_price.values,
                        y=state_price.index,
                        orientation='h',
                        title="Average Prices by State",
                        color=state_price.values,
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 🏪 Market Analysis")
                    top_markets = filtered_df.groupby('market')['price'].mean().sort_values(ascending=False).head()

                    fig = px.bar(
                        x=top_markets.index,
                        y=top_markets.values,
                        title="Top 10 Markets by Average Price",
                        color=top_markets.values,
                        color_continuous_scale='plasma'
                    )
                    fig.update_xaxes(tickangle=45)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                #summary statistics
                st.markdown("### 📊 Summary Statistics")

                summary_stats = filtered_df.groupby('commodity').agg({
                    'price': ['count', 'mean', 'median', 'std', 'min', 'max'],
                    'usdprice': ['mean', 'median']
                }).round(2)

                summary_stats.columns = ['Records', 'Mean_NGN', 'Median_NGN', 'StdDev_NGN', 'Min_NGN', 'Max_NGN', 'Mean_USD', 'Median_USD']

                st.dataframe(summary_stats, use_container_width=True)
            else:
                st.warning("No data available for the selected filters. Please adjust your selections.")
        else:
            st.info("Please select at least one commodity and one state to view the analysis.")
    else:
        st.info("Please upload your dataset to get started!")
        st.markdown("""
        ### 📁 Getting Started
        
        1. **Prepare your data**: Ensure your CSV file contains the required columns:
           - `date`, `admin1`, `admin2`, `market`, `commodity`, `price`, `usdprice`, `latitude`, `longitude`
        
        2. **Place the file**: Put your CSV file at:
           ```
           data/raw/nigerian_food_prices_2002_2025.csv
           ```
        
        3. **Load and clean**: Use the sidebar controls to load and clean your data
        
        4. **Explore**: Navigate through different pages for detailed analysis
        """)

if __name__ == "__main__":
    main()


