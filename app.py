import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="California Housing Price Predictor", layout="wide")

# Load model and feature names
model = joblib.load("model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("California Housing Price Predictor")
st.markdown(
    """
    This app predicts California housing prices using a Random Forest model.
    Adjust the housing and location characteristics in the sidebar to generate a prediction.
    """
)

st.sidebar.header("Input Features")

# Sidebar inputs based on California Housing dataset
medinc = st.sidebar.slider("Median Income", min_value=0.0, max_value=15.0, value=4.0, step=0.1)
houseage = st.sidebar.slider("House Age", min_value=1.0, max_value=52.0, value=25.0, step=1.0)
averooms = st.sidebar.slider("Average Rooms", min_value=1.0, max_value=15.0, value=5.0, step=0.1)
avebedrms = st.sidebar.slider("Average Bedrooms", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
population = st.sidebar.number_input("Population", min_value=1.0, max_value=40000.0, value=1500.0, step=100.0)
aveoccup = st.sidebar.slider("Average Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
latitude = st.sidebar.slider("Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.1)
longitude = st.sidebar.slider("Longitude", min_value=-125.0, max_value=-114.0, value=-118.0, step=0.1)

# Input dataframe
input_data = pd.DataFrame([{
    "MedInc": medinc,
    "HouseAge": houseage,
    "AveRooms": averooms,
    "AveBedrms": avebedrms,
    "Population": population,
    "AveOccup": aveoccup,
    "Latitude": latitude,
    "Longitude": longitude
}])

# Reorder columns just in case
input_data = input_data[feature_names]

# Prediction
prediction = model.predict(input_data)[0]

# Approximate uncertainty band
lower_bound = prediction - 0.50
upper_bound = prediction + 0.50

st.subheader("Prediction")
st.metric("Predicted Housing Price", f"{prediction:.3f}")
st.write(f"Approximate prediction interval: **[{lower_bound:.3f}, {upper_bound:.3f}]**")

st.subheader("Input Summary")
st.dataframe(input_data)

# Simple updating visualization
st.subheader("Feature Values")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(input_data.columns, input_data.iloc[0].values)
ax.set_title("Selected Input Features")
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig)

st.markdown(
    """
    **Important:** This app provides a predictive estimate only.  
    It does **not** identify causal effects.
    """
)