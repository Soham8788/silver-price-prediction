import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import pickle
import numpy as np

st.set_page_config(page_title="Silver Price Predictor", layout="wide")

st.title("📈 Silver Price Prediction Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Controls")
days_to_show = st.sidebar.slider("Days of history to show", 30, 730, 365)

@st.cache_data
def load_data():
    """Load silver price data"""
    try:
        # Try to load from CSV first
        df = pd.read_csv('data/silver_data.csv', parse_dates=['Date'])
        
        # Ensure numeric columns
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        return df
    except:
        # Fallback to yfinance
        silver = yf.download('SLV', period='2y', progress=False, auto_adjust=False)
        if isinstance(silver.columns, pd.MultiIndex):
            silver.columns = [col[0] for col in silver.columns]
        return silver

@st.cache_resource
def load_model():
    try:
        with open('models/silver_price_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load data
with st.spinner("Loading market data..."):
    data = load_data()

# Make sure we have numeric values (convert Series to scalar)
current_price = float(data['Close'].iloc[-1])
prev_price = float(data['Close'].iloc[-2])
change = current_price - prev_price
change_pct = (change / prev_price) * 100

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰 Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")

with col2:
    high_52w = float(data['High'].max())
    st.metric("📈 52-Week High", f"${high_52w:.2f}")

with col3:
    low_52w = float(data['Low'].min())
    st.metric("📉 52-Week Low", f"${low_52w:.2f}")

with col4:
    volume = float(data['Volume'].iloc[-1])
    st.metric("📊 Volume", f"{volume:,.0f}")

st.markdown("---")

# Price chart
st.subheader("📊 Silver Price History")

# Filter data for selected days
data_filtered = data.tail(days_to_show)

fig = go.Figure()

# Main price line
fig.add_trace(go.Scatter(
    x=data_filtered.index,
    y=data_filtered['Close'],
    mode='lines',
    name='Silver Price',
    line=dict(color='blue', width=2)
))

# 20-day moving average
ma20 = data_filtered['Close'].rolling(window=20).mean()
fig.add_trace(go.Scatter(
    x=data_filtered.index,
    y=ma20,
    mode='lines',
    name='20-Day MA',
    line=dict(color='orange', width=1.5, dash='dash')
))

# 50-day moving average
ma50 = data_filtered['Close'].rolling(window=50).mean()
fig.add_trace(go.Scatter(
    x=data_filtered.index,
    y=ma50,
    mode='lines',
    name='50-Day MA',
    line=dict(color='green', width=1.5, dash='dash')
))

fig.update_layout(
    title='Silver Price with Moving Averages',
    xaxis_title='Date',
    yaxis_title='Price ($ per ounce)',
    hovermode='x unified',
    height=500,
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

# Volume chart
st.subheader("📊 Trading Volume")

fig_volume = go.Figure()
fig_volume.add_trace(go.Bar(
    x=data_filtered.index,
    y=data_filtered['Volume'],
    name='Volume',
    marker_color='lightblue',
    opacity=0.7
))

fig_volume.update_layout(
    title='Daily Trading Volume',
    xaxis_title='Date',
    yaxis_title='Volume (shares)',
    height=300,
    template='plotly_white'
)

st.plotly_chart(fig_volume, use_container_width=True)

# Prediction section
st.markdown("---")
st.subheader("🔮 Price Prediction")

model_data = load_model()

if model_data:
    st.success(f"✅ Model loaded: **{model_data['model_name']}**")
    st.info(f"📈 Model Performance: R² Score = {model_data['results']['R2']:.4f}")
    st.caption(f"💰 Prediction Error (RMSE): ${model_data['results']['RMSE']:.2f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Generate Prediction", use_container_width=True):
            st.info("To make predictions, use the prediction script or train with latest data")
            st.code("python test_prediction.py", language="bash")
    
    with col2:
        # Show last known price
        st.metric("Last Known Price", f"${current_price:.2f}")
        
else:
    st.error("❌ Model not found. Please train the model first:")
    st.code("python src/train_model.py", language="bash")

# Statistics section
st.markdown("---")
st.subheader("📊 Price Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    mean_price = float(data['Close'].mean())
    st.metric("Average Price (All Time)", f"${mean_price:.2f}")

with col2:
    std_price = float(data['Close'].std())
    st.metric("Price Volatility (Std Dev)", f"${std_price:.2f}")

with col3:
    total_return = ((current_price - float(data['Close'].iloc[0])) / float(data['Close'].iloc[0])) * 100
    st.metric("Total Return", f"{total_return:+.1f}%")

# Footer
st.markdown("---")
st.caption("📊 Data source: Yahoo Finance (SLV - iShares Silver Trust)")
st.caption("🔧 Built with Streamlit, yfinance, and scikit-learn")
st.caption(f"📅 Data last updated: {data.index[-1].strftime('%Y-%m-%d')}")