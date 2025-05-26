import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("E:/College/Research/bot_2/forecaster-23f9d-firebase-adminsdk-fbsvc-3e068ad9d4.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def fetch_historical_data():
    """Fetch historical tourism data from Firebase."""
    historical_data = db.collection("historical_tourism").stream()
    data_list = []
    for doc in historical_data:
        data = doc.to_dict()
        data_list.append(data)
    return pd.DataFrame(data_list)

def train_and_save_models():
    """Train models and save them as .pkl files."""
    df = fetch_historical_data()
    
    if df.empty or len(df) < 3:
        print("Not enough data to train models. At least 3 records required.")
        return

    # Prepare features with one-hot encoding for season
    encoder = OneHotEncoder(sparse_output=False, categories=[["Spring", "Summer", "Autumn", "Winter"]])
    season_encoded = encoder.fit_transform(df[['season']])
    season_df = pd.DataFrame(season_encoded, columns=encoder.get_feature_names_out(['season']))

    X = pd.concat([season_df, df[['year']].reset_index(drop=True)], axis=1)
    y_tourists = df['historical_tourists']
    y_spending = df['avg_spending']
    y_flights = df['flight_arrivals']
    y_occupancy = df['hotel_occupancy']

    # Train models
    models = {
        "Tourists": LinearRegression().fit(X, y_tourists),
        "Spending": LinearRegression().fit(X, y_spending),
        "Flight Arrivals": LinearRegression().fit(X, y_flights),
        "Hotel Occupancy": LinearRegression().fit(X, y_occupancy)
    }

    # Save models and encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoder, "models/season_encoder.pkl")
    for name, model in models.items():
        joblib.dump(model, f"models/{name.lower().replace(' ', '_')}_model.pkl")

    print("Models and encoder saved successfully!")

if __name__ == "__main__":
    train_and_save_models()