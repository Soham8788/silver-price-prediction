import pandas as pd
import os

print("Cleaning up silver_data.csv...")

# Read the file
df = pd.read_csv('data/silver_data.csv')

# Check if columns are tuples (like ('Date', ''), ('Close', 'SLV'))
if isinstance(df.columns, pd.MultiIndex) or isinstance(df.columns[0], tuple):
    # Flatten the column names
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            # Take the first element of the tuple
            new_col = col[0]
            new_columns.append(new_col)
        else:
            new_columns.append(col)
    df.columns = new_columns

# Remove any duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# Ensure Date column is properly formatted
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Make sure numeric columns are numeric
numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows with NaN in Close
df = df.dropna(subset=['Close'])

# Add Adj Close if missing
if 'Adj Close' not in df.columns:
    df['Adj Close'] = df['Close']

# Save cleaned data
df.to_csv('data/silver_data.csv', index=False)

print(f"\n✅ Data cleaned!")
print(f"📊 Shape: {df.shape}")
print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"💰 Close price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
print(f"\n📋 Columns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3))