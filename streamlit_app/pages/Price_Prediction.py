import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime


# Base project directory (nigeria_food_prices/)
BASE_DIR = Path(__file__).parent.parent

# Add project root (for config.py, etc.)
sys.path.append(str(BASE_DIR))

# Add src directory (for your reusable modules)
sys.path.append(str(BASE_DIR / "src"))

@st.cache_resource
def load_model():
    # data_path = BASE_DIR / "models" / "saved_models" / "food_price_model.pkl"
    data_path = "models/saved_models/food_price_model.pkl"
    return joblib.load(data_path)

@st.cache_data
def load_data():
    # data_path = BASE_DIR / "data" / "raw" / "nigerian_food_prices_2002_2025.csv"
    # df = pd.read_csv(data_path)
    df = st.session_state.cleaned_data
    return df

def user_input_form(df):
    st.sidebar.header("🔮 Prediction Input")
    category = st.sidebar.selectbox("Select Category", sorted(df['category'].unique()))
    commodity = st.sidebar.selectbox("Select Commodity", sorted(df["commodity"].unique()))
    admin1 = st.sidebar.selectbox("Select State", sorted(df["admin1"].unique()))
    market = st.sidebar.selectbox("Select Market", sorted(df["market"].unique()))
    unit = st.sidebar.selectbox("Select Unit", sorted(df["unit"].unique()))
    pricetype = st.sidebar.selectbox("Price Type", sorted(df["pricetype"].unique()))

    date = st.sidebar.date_input("Select Date", datetime(2025, 1, 1))

    # latitude = st.sidebar.number_input("Latitude", value=float(df["latitude"].mean()))
    # longitude = st.sidebar.number_input("Longitude", value=float(df["longitude"].mean()))

    features = pd.DataFrame({
        "date": [date],
        "admin1": [admin1],
        "market": [market],
        # "latitude": [latitude],
        # "longitude": [longitude],
        "category": [category],
        "commodity": [commodity],
        "unit": [unit],
        # "priceflag": ["Actual"],
        "pricetype": [pricetype],
        "currency": ["NGN"],
    })

    return features

def make_prediction(model_bundle, input_df):
    model, preprocessor = model_bundle["model"], model_bundle["preprocessor"]
    X = preprocessor.transform(input_df)
    prediction = model.predict(X)[0]

    # # Clamp prediction within observed price range
    # min_price, max_price = df["price"].min(), df["price"].max()
    # prediction = np.clip(raw_pred, min_price, max_price)

    return prediction


def price_trend_chart(df, commodity):
    df_filtered = df[df["commodity"] == commodity]
    fig = px.line(df_filtered, x="date", y="price", color="market",
                  title=f"📈 Price Trend for {commodity}",
                  labels={"price": "Price (NGN)", "date": "Date"})
    
    return fig

def commodity_comparison_chart(df, state):
    df_filtered = df[df["admin1"] == state].groupby(["commodity"], as_index=False)["price"].mean()
    fig = px.bar(df_filtered, x="commodity", y="price", color="commodity",
                 title=f"🏙️ Average Commodity Prices in {state}",
                 labels={"price": "Avg Price (NGN)"})
    
    return fig

def market_heatmap(df):
    df_grouped = df.groupby(["market", "latitude", "longitude"], as_index=False)["price"].mean()
    fig = px.scatter_mapbox(df_grouped, lat="latitude", lon="longitude", size="price",
                            color="price", hover_name="market",
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            title="🗺️ Market Price Heatmap",
                            mapbox_style="carto-positron", zoom=5, height=500)
    
    return fig

def main():
    st.set_page_config(
        page_title="Nigeria Food Price Prediction",
        page_icon="🥘",
        layout="wide"
    )

    st.title("🥘 Nigeria Food Price Prediction Dashboard")
    st.markdown("An advance app to **analyze trends** and **predict food prices** across Nigeria.")
    #check for data
    if not hasattr(st.session_state, 'cleaned_data'):
        st.warning("No cleaned data found. Please return to the main page and clean the dataset first.")
        return
    

    #load resources
    model_bundle = load_model()
    df = load_data()

    #sidebar input
    user_input = user_input_form(df)

    #make prediction
    if st.sidebar.button("🔮 Predict Price"):
        prediction = make_prediction(model_bundle, user_input)
        st.success(f"### 💰 Predicted Price: **₦{prediction:,.2f}**") 

        # # Optional: Show debugging info
        # with st.expander("ℹ️ Prediction Details"):
        #     st.write(f"Raw prediction: ₦{raw_pred:,.2f}")
        #     st.write(f"Clamped range: ₦{min_price:,.2f} – ₦{max_price:,.2f}")
    
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(price_trend_chart(df, user_input["commodity"][0]), use_container_width=True)
    
    with col2:
        st.plotly_chart(commodity_comparison_chart(df, user_input["admin1"][0]), use_container_width=True)
    
    st.plotly_chart(market_heatmap(df), use_container_width=True)

    st.markdown('---')
    st.caption('Built with ❤️ - All rights reserved © Lukman Olamide Ademola')

if __name__ == "__main__":
    main()