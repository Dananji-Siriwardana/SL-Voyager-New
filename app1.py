import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from time import sleep
import joblib
import os
import requests  # For Flask authentication check

# Custom CSS (unchanged, but adjusted colors to match Flask’s #00aaff)
st.markdown("""
    <style>
    .main {
        background: #1a1a2e;  /* Match Flask background */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background: linear-gradient(to right, #00aaff, #66ccff);  /* Match Flask primary color */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
        border: 1px solid #00aaff;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #66ccff, #99ddff);
        transform: scale(1.05);
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .stSelectbox, .stNumberInput {
        background-color: rgba(255, 255, 255, 0.1);  /* Translucent like Flask cards */
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #00aaff;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #00aaff;  /* Match Flask text-primary */
        font-family: 'Arial', sans-serif;
        text-shadow: 1px 1px 2px rgba(0, 105, 255, 0.3);
    }
    .stDataFrame {
        border: 2px solid #00aaff;
        border-radius: 8px;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.2);  /* Match Flask card style */
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #00aaff;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Check Flask authentication (optional)
def check_auth():
    try:
        response = requests.get("http://localhost:5000/check_auth", cookies=st.session_state.get("cookies", {}))
        if response.status_code == 200 and response.json().get("authenticated"):
            return True
        return False
    except:
        return False

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("forecaster-23f9d-firebase-adminsdk-fbsvc-3e068ad9d4.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Load models and encoder
try:
    encoder = joblib.load("models/season_encoder.pkl")
    models = {
        "Tourists": joblib.load("models/tourists_model.pkl"),
        "Spending": joblib.load("models/spending_model.pkl"),
        "Flight Arrivals": joblib.load("models/flight_arrivals_model.pkl"),
        "Hotel Occupancy": joblib.load("models/hotel_occupancy_model.pkl")
    }
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py to generate models.")
    st.stop()

# Authentication check
if not check_auth():
    st.warning("Please log in to SL-VOYAGER to access the forecasting feature.")
    login_url = "http://localhost:5000/login"
    st.markdown(f"[Log in here]({login_url})")
    st.stop()

# Rest of your Streamlit code (unchanged)
st.title("🌊 Tourism Spending Forecaster")
# ... (sidebar, tabs, historical data, predictions, trends, delete data sections remain the same)