import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from preprocess import create_features, prepare_data

def train_models(X_train, X_test, y_train, y_test, scaler_y):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = -np.inf
    best_name = ""
    
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    for name, model in models.items():
        print(f"\n📊 Training {name}...")
        model.fit(X_train, y_train.ravel())
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_actual = scaler_y.inverse_transform(y_test)
        
        r2 = r2_score(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'model': model}
        print(f"   R²: {r2:.4f}, RMSE: ${rmse:.2f}, MAE: ${mae:.2f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
    
    return best_model, best_name, results

if __name__ == "__main__":
    print("\n🚀 Silver Price Prediction Pipeline")
    print("="*50)
    
    df = pd.read_csv('data/silver_data.csv', parse_dates=['Date'])
    df_featured = create_features(df)
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_cols = prepare_data(df_featured)
    
    best_model, best_name, results = train_models(X_train, X_test, y_train, y_test, scaler_y)
    
    os.makedirs('models', exist_ok=True)
    with open('models/silver_price_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'model_name': best_name,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_cols': feature_cols,
            'results': results[best_name]
        }, f)
    
    print("\n" + "="*50)
    print("✅ FINAL RESULTS")
    print("="*50)
    print(f"🏆 Best Model: {best_name}")
    print(f"📈 R² Score: {results[best_name]['R2']:.4f}")
    print(f"💰 RMSE: ${results[best_name]['RMSE']:.2f}")
    print(f"\n💾 Model saved to models/silver_price_model.pkl")