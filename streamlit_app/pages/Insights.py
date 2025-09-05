import io
from datetime import datetime
from functools import lru_cache
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from scipy import stats
from statsmodels.tsa.seasonal import STL

def preprocess(df):
    df = df.copy()
    #normalize column names
    df.columns = [c.strip() for c in df.columns]

    #parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    #numeric conversions
    for col in ['price', 'usdprice', 'latitude', 'market_id', 'commodity_id']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    #fill categorical NAs
    for c in ['admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'currency', 'pricetype', 'priceflag']:
        if c in df.columns:
            df[c] = df[c].astype('object').fillna('Unknown')
    
    #add year, month for grouping
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    return df

def generate_summary(df, group_by=None):
    metrics = {}
    metrics['rows'] = len(df)
    metrics['commodities'] = df['commodity'].nunique() if 'commodity' in df.columns else 0
    metrics['markets'] = df['market'].nunique() if 'market' in df.columns else 0
    if 'price' in df.columns:
        metrics['price_mean'] = float(df['price'].mean())
        metrics['price_median'] = float(df['price'].median())
        metrics['price_std'] = float(df['price'].std())
    if group_by and group_by in df.columns:
        grp = df.groupby(group_by)['price'].agg(['mean', 'median', 'count', 'std']).reset_index()
        return metrics, grp.sort_values('count', ascending=False)
    return metrics, None

def compute_price_trend(df, commodity=None, market=None, window=3):
    d = df.copy()
    if commodity:
        d = d[d['commodity'] == commodity]
    if market:
        d = d[d['market'] == market]
    if 'year_month' not in d.columns:
        return pd.DataFrame()
    ts = d.groupby('year_month')['price'].mean().reset_index().sort_values('year_month')
    ts['rolling_mean'] = ts['price'].rolling(window=window, min_periods=1).mean()
    ts['pct_change'] = ts['rolling_mean'].pct_change()
    ts['trend_direction'] = np.where(ts['pct_change'] > 0, 'up', np.where(ts['pct_change'] < 0, 'down', 'flat'))
    return ts
    
def detect_anomalies(ts, z_thresh=3, window=12):
    t = ts.copy()
    if t.empty:
        return t
    t = t.set_index('year_month')
    rolling_mean = t['price'].rolling(window=window, min_periods=1).mean()
    rolling_std = t['price'].rolling(window=window, min_periods=1).std().fillna(0.0)
    z = (t['price'] - rolling_mean) / (rolling_std.replace(0, np.nan))
    t = t.reset_index()
    t['z_score'] = z.values
    t['anomaly'] = t['z_score'].abs() > z_thresh
    return t

def seasonal_decompose(df_ts, period=12):
    if df_ts.empty or len(df_ts) < period*2:
        return None
    try:
        stl = STL(df_ts['price'].values, period=period, robust=True)
        res = stl.fit()
        return pd.DataFrame({'trend':res.trend, 'seasonal':res.seasonal, 'resid':res.resid}, index=df_ts['year_month'].values)
    except Exception:
        return None

def top_n(df, by='market', n=10):
    if by not in df.columns or 'price' not in df.columns:
        return pd.DataFrame()
    return df.groupby(by)['price'].agg(['mean', 'median', 'count']).reset_index().sort_values('mean', ascending=False).head(n)

def correlation_matrix(df, pivot_index='market', pivot_columns='commodity'):
    try:
        pivot = df.pivot_table(index=pivot_index, columns=pivot_columns, values='price', aggfunc='mean')
        corr = pivot.corr()
        return corr, pivot
    except Exception:
        return None, None
    
def cluster_commodities(df, n_clusters=3):
    pivot = df.pivot_table(index='commodity', columns='market', values='price', aggfunc='mean').fillna(0)
    if pivot.shape[0] < n_clusters:
        return None
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pivot.values)
    out = pd.DataFrame({'commodity': pivot.index, 'cluster':labels})
    return out, pivot

def main():
    st.set_page_config(
        page_title="Automated Insights - Market Prices",
        page_icon="💡",
        layout="wide"
    )
    st.title("📊 Automated Market Price Insights")
    st.markdown('This app generates advanced insights')

    if not hasattr(st.session_state, 'cleaned_data'):
        st.warning("No cleaned data found. Please return to the main page and clean the dataset first.")
        return
    
    raw_df = st.session_state.cleaned_data
    df = preprocess(raw_df)

    #quick preview
    with st.expander('Data Preview'):
        st.dataframe(df.head(50))
    
    #filters
    st.sidebar.header("Filters")
    if 'date' in df.columns:
        min_date = df['date'].min()
        max_date = df['date'].max()
        date_range = st.sidebar.date_input('Date range', value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        date_range = None
    
    admin1_list = sorted(df['admin1'].dropna().unique().tolist()) if 'admin1' in df.columns else []
    admin1_choice = st.sidebar.multiselect('State (admin1)', options=admin1_list, default=None)


    market_list = sorted(df['market'].dropna().unique().tolist()) if 'market' in df.columns else []
    market_choice = st.sidebar.multiselect('Market', options=market_list, default=None)


    category_list = sorted(df['category'].dropna().unique().tolist()) if 'category' in df.columns else []
    category_choice = st.sidebar.multiselect('Category', options=category_list, default=None)


    commodity_list = sorted(df['commodity'].dropna().unique().tolist()) if 'commodity' in df.columns else []
    commodity_choice = st.sidebar.selectbox('Commodity (for detailed trend)', options=[None] + commodity_list, index=0)

    #apply filters
    fdf = df.copy()
    if date_range and 'date' in df.columns:
        start, end = date_range
        fdf = fdf[(fdf['date'] >= pd.to_datetime(start)) & (fdf['date'] <= pd.to_datetime(end))]
    if admin1_choice:
        fdf = fdf[fdf['admin1'].isin(admin1_choice)]
    if market_choice:
        fdf = fdf[fdf['market'].isin(market_choice)]
    if category_choice:
        fdf = fdf[fdf['category'].isin(category_choice)]
    
    #insights
    summary_metrics, grouped_by_market = generate_summary(fdf, group_by='market')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Rows', summary_metrics.get('rows', 0))
    col2.metric('Unique Commodities', summary_metrics.get('markets', 0))
    col3.metric('Unique Markets', summary_metrics.get('markets', 0))
    col4.metric('Avg Price', f"{summary_metrics.get('price_mean', 0):,.2f}")

    with st.expander("Top markets by mean price"):
        top_markets = top_n(fdf, by='market', n=15)
        st.dataframe(top_markets)
        if not top_markets.empty:
            fig = px.bar(top_markets, x='market', y='mean', hover_data=['median', 'count'], title='Top markets by mean price')
            st.plotly_chart(fig, use_container_width=True)
    
    #commodity / market time series
    st.markdown('---')
    st.header('Time Series & Anomaly Detection')

    selected_commodity = commodity_choice
    selected_market = market_choice[0] if market_choice else None
    window = st.slider('Smoothing window (months)', min_value=1, max_value=24, value=3)

    ts = compute_price_trend(fdf, commodity=selected_commodity, market=selected_market, window=window)
    if ts.empty:
        st.info("No time series available for the chosen filters.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts['year_month'], y=ts['price'], mode='lines+markers', name='mean_price'))
        fig.add_trace(go.Scatter(x=ts['year_month'], y=ts['rolling_mean'], mode='lines', name=f'rolling_mean_{window}'))
        fig.update_layout(title=f'Time Series for {selected_commodity or "All Commodities"} {"-"+selected_market if selected_market else ""}', xaxis_title="Date", yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)

        #anomalies
        z_thresh = st.slider("Z-score threshold for anomaly", min_value=1.0, max_value=5.0, value=3.0)
        anomalies = detect_anomalies(ts, z_thresh=z_thresh, window=12)
        if not anomalies.empty:
            anom_points = anomalies[anomalies['anomaly']]
            if not anom_points.empty:
                st.markdown('**Detected anomalies**')
                st.dataframe(anom_points[['year_month', 'price', 'z_score']])
                fig2 = px.scatter(anom_points, x='year_month', y='price', size=anom_points['z_score'].abs(), title="Anomalies (size ~ |z-score|)")
                st.plotly_chart(fig2, use_container_width=True)
        
        #seasonality overview
        if len(ts) >= 24:
            res = seasonal_decompose(ts, period=12)
            if res is not None:
                st.subheader("STL Decomposition (trend / seasonal / residual)")
                df_res = res.reset_index().rename(columns={'index': 'date'})
                st.plotly_chart(px.line(df_res, x='date', y='trend', title='Trend'), use_container_width=True)
                st.plotly_chart(px.line(df_res, x='date', y='seasonal', title='Seasonal'), use_container_width=True)
                st.plotly_chart(px.line(df_res, x='date', y='resid', title='Residual'), use_container_width=True)
    
    #correlation & clustering
    st.markdown('---')
    st.header("Correlaation & Clustering")

    corr, pivot = correlation_matrix(fdf, pivot_index='market', pivot_columns='commodity')
    if corr is not None:
        st.subheader("Commodity correlation matrix")
        fig_corr = px.imshow(
            corr, 
            labels=dict(x='commodity', y='commodity', color='correlation'), 
            x=corr.columns, 
            y=corr.index, 
            title='Commodity Price Correlation',
            # color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            # text_auto=True
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    cluster_count = st.slider('Commodity cluster count (KMeans)', min_value=2, max_value=8, value=3)
    cluster_res = cluster_commodities(fdf, n_clusters=cluster_count)
    if cluster_res is not None:
        out, pivot_mat = cluster_res
        st.subheader('Commodity clusters')
        st.dataframe(out.sort_values('cluster'))
        cluster_sizes = out.groupby('cluster').size().reset_index(name='n_commodities')
        st.bar_chart(cluster_sizes.set_index('cluster'))
    

    #geographic view
    st.markdown("---")
    st.header("Map - Market locations")
    if 'latitude' in fdf.columns and 'longitude' in fdf.columns:
        map_df = fdf.dropna(subset=['latitude', 'longitude']).drop_duplicates(['market', 'latitude', 'longitude'])
        if not map_df.empty:
            st.map(map_df[['latitude', 'longitude']])
        else:
            st.info("No geolocation data to plot.")
    
    #export insights
    st.markdown('---')
    st.header("Export Insights")

    insights = {}
    insights['summary'] = summary_metrics
    if grouped_by_market is not None:
        insights['market_summary'] = grouped_by_market
    if not ts.empty:
        insights['time_series'] = ts
    if corr is not None:
        insights['correlation'] = corr
    if cluster_res is not None:
        insights['clusters'] = out
    
    if st.button("Generate downloadable insights (Excel)"):
        with st.spinner('Preparing Excel file...'):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                pd.DataFrame([summary_metrics]).to_excel(writer, sheet_name='summary', index=False)
                if grouped_by_market is not None:
                    grouped_by_market.to_excel(writer, sheet_name='market_summary', index=False)
                if not ts.empty:
                    ts.to_excel(writer, sheet_name='time_series', index=False)
                if corr is not None:
                    corr.to_excel(writer, sheet_name='correlation')
                if cluster_res is not None:
                    out.to_excel(writer, sheet_name='cluster', index=False)
            output.seek(0)
            st.download_button('Download Insight Excel', data=output, file_name='Nigeria_food_Insights.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    st.markdown('---')
    st.caption('Built with ❤️ - All rights reserved © Lukman Olamide Ademola')

if __name__ == "__main__":
    main()














