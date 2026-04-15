import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_features(df):
    df = df.copy()
    df['MA_7'] = df['Close'].rolling(7).mean()
    df['MA_21'] = df['Close'].rolling(21).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(10).std()
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    
    for lag in [1, 2, 3]:
        df[f'Lag_{lag}'] = df['Close'].shift(lag)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    print(f"✅ Features created. Shape: {df.shape}")
    return df

def prepare_data(df, target_col='Close', test_size=0.2):
    exclude_cols = ['Date', target_col, 'Open', 'High', 'Low', 'Adj Close', 'Volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y, feature_cols

if __name__ == "__main__":
    df = pd.read_csv('data/silver_data.csv', parse_dates=['Date'])
    df_featured = create_features(df)
    prepare_data(df_featured)