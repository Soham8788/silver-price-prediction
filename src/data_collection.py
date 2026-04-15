import yfinance as yf
from datetime import datetime
import os
import pandas as pd

def fetch_silver_data(start_date='2015-01-01', end_date=None):
    """
    Fetch historical silver price data from Yahoo Finance
    Using SLV (iShares Silver Trust ETF)
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching silver data from {start_date} to {end_date}...")
    
    # Download SLV data
    silver = yf.download('SLV', start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if len(silver) == 0:
        print("❌ Failed to fetch data")
        return None
    
    # Reset index to make Date a column
    silver.reset_index(inplace=True)
    
    # Flatten column names if they are MultiIndex
    if isinstance(silver.columns, pd.MultiIndex):
        silver.columns = ['_'.join(col).strip() for col in silver.columns.values]
    
    # Rename columns to simple names
    column_mapping = {}
    for col in silver.columns:
        if 'Date' in col or 'date' in str(col).lower():
            column_mapping[col] = 'Date'
        elif 'Close' in col:
            column_mapping[col] = 'Close'
        elif 'High' in col:
            column_mapping[col] = 'High'
        elif 'Low' in col:
            column_mapping[col] = 'Low'
        elif 'Open' in col:
            column_mapping[col] = 'Open'
        elif 'Volume' in col:
            column_mapping[col] = 'Volume'
    
    silver.rename(columns=column_mapping, inplace=True)
    
    # Ensure we have the required columns
    required_cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    for col in required_cols:
        if col not in silver.columns:
            print(f"⚠️ Missing column: {col}")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    silver.to_csv('data/silver_data.csv', index=False)
    
    print(f"\n✅ Data saved to data/silver_data.csv")
    print(f"📊 Shape: {silver.shape}")
    print(f"📅 Date range: {silver['Date'].min()} to {silver['Date'].max()}")
    print(f"📋 Columns: {silver.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(silver.head(3))
    
    return silver

if __name__ == "__main__":
    fetch_silver_data()