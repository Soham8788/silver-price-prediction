import pickle
import pandas as pd
import numpy as np

def load_model():
    try:
        with open('models/silver_price_model.pkl', 'rb') as f:
            data = pickle.load(f)
        print("✅ Model loaded successfully!")
        return data
    except FileNotFoundError:
        print("❌ Model not found. Run src/train_model.py first.")
        return None

def predict_price(features_dict):
    data = load_model()
    if data is None:
        return None
    
    model = data['model']
    scaler_X = data['scaler_X']
    scaler_y = data['scaler_y']
    feature_cols = data['feature_cols']
    
    features_df = pd.DataFrame([features_dict])
    missing = set(feature_cols) - set(features_df.columns)
    if missing:
        print(f"❌ Missing features: {missing}")
        return None
    
    features_scaled = scaler_X.transform(features_df[feature_cols])
    pred_scaled = model.predict(features_scaled)
    prediction = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    
    return prediction[0][0]

if __name__ == "__main__":
    data = load_model()
    if data:
        print(f"📊 Model: {data['model_name']}")
        print(f"📈 R² Score: {data['results']['R2']:.4f}")