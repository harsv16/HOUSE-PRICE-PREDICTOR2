import streamlit as st
import pandas as pd
import pickle

# --------------------------
# 🔧 Utility Functions (from utils.py)
# --------------------------

def load_model(path='model.pkl'):
    """Loads the trained model from a pickle file."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Make sure 'model.pkl' exists.")
        return None

def preprocess_input(location, area, bedrooms, bathrooms, age):
    """Formats user input into a DataFrame for prediction."""
    return pd.DataFrame({
        'Location': [location],
        'Area (sqft)': [area],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Age': [age]
    })

# --------------------------
# 🖥 Streamlit App
# --------------------------

# Set page config
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

# Title
st.title("🏠 House Price Prediction App")
st.markdown("Enter the property details below to estimate its market value.")

# User Inputs
location = st.selectbox("📍 Location", [
    "Mumbai", "Bengaluru", "Delhi", "Pune", "Hyderabad",
    "Chennai", "Ahmedabad", "Kolkata", "Jaipur", "Lucknow"
])

area = st.number_input("📐 Area (in sqft)", min_value=300, max_value=10000, step=50)
bedrooms = st.selectbox("🛏 Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("🛁 Bathrooms", [1, 2, 3, 4])
age = st.slider("🏚 Property Age (Years)", 0, 50, step=1)

# Load model
model = load_model("model.pkl")

# Prediction
if st.button("🔍 Predict Price"):
    if model is not None:
        input_df = preprocess_input(location, area, bedrooms, bathrooms,age)
        prediction = model.predict(input_df)
        st.success(f"estimated House Price:{round(prediction[0],2)} lakhs")
        
