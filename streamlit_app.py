#Step 6: 🎛️ Create the Streamlit Web App

# app.py
import streamlit as st
import pickle
import numpy as np

with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)
st.title("🚗 Vehicle Price Prediction")
horsepower = st.number_input("Horsepower", min_value=0, value=300)
torque = st.number_input("Torque", min_value=0, value=400)
if st.button("Predict Price"):
    features = np.array([[horsepower, torque]])
    price = model.predict(features)
    st.success(f"Estimated Price: ${price[0]:,.2f}")
