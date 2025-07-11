import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# --------------------------
# 🔧 Utility Functions
# --------------------------
def load_model(path='model.pkl'):
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Make sure 'model.pkl' exists.")
        return None

def preprocess_input(location, area, bedrooms, bathrooms, age):
    return pd.DataFrame({
        'Location': [location],
        'Area (sqft)': [area],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Age': [age]
    })

# --------------------------
# 🎨 Styling
# --------------------------
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

st.markdown("""
    <style>
        .main-title {
            color: #1E90FF;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
        }
        .result-box {
            background-color: #e6f7ff;
            padding: 20px;
            border-radius: 10px;
            color: #003366;
            text-align: center;
            font-size: 24px;
            margin-top: 20px;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 14px;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# 🖼 Header
# --------------------------
st.image("https://img.icons8.com/clouds/500/real-estate.png", width=100)
st.markdown("<div class='main-title'>🏠 House Price Predictor</div><br>", unsafe_allow_html=True)
st.markdown("Enter the property details below to estimate its market value.")

# --------------------------
# 📝 User Inputs
# --------------------------
location = st.selectbox("📍 Location", [
    "Mumbai", "Bengaluru", "Delhi", "Pune", "Hyderabad",
    "Chennai", "Ahmedabad", "Kolkata", "Jaipur", "Lucknow"
])

area = st.number_input("📐 Area (in sqft)", min_value=300, max_value=10000, step=50)
bedrooms = st.selectbox("🛏 Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("🛁 Bathrooms", [1, 2, 3, 4])
age = st.slider("🏚 Property Age (Years)", 0, 50, step=1)

# --------------------------
# 🧠 Load Model
# --------------------------
model = load_model("model.pkl")

# --------------------------
# 🔍 Prediction
# --------------------------
if st.button("🔍 Predict Price"):
    if model is not None:
        input_df = preprocess_input(location, area, bedrooms, bathrooms, age)
        prediction = model.predict(input_df)[0]
        st.markdown(f"""
        <div class='result-box'>
            💰 Estimated House Price: ₹ {round(prediction, 2)} lakhs
        </div>
        """, unsafe_allow_html=True)

        # --------📈 Optional Chart--------
        area_range = np.arange(500, 5001, 500)
        predicted_prices = []
        for a in area_range:
            test_df = preprocess_input(location, a, bedrooms, bathrooms, age)
            price = model.predict(test_df)[0]
            predicted_prices.append(price)

        fig, ax = plt.subplots()
        ax.plot(area_range, predicted_prices, marker='o', color='green')
        ax.set_title("Area vs Predicted Price")
        ax.set_xlabel("Area (sq.ft)")
        ax.set_ylabel("Price (₹ lakhs)")
        st.pyplot(fig)

# --------------------------
# 📂 Optional: Upload CSV
# --------------------------
st.markdown("### 📂 Upload CSV for Batch Prediction")
csv_file = st.file_uploader("Upload CSV file", type="csv")
if csv_file:
    df = pd.read_csv(csv_file)
    required = {'Location', 'Area (sqft)', 'Bedrooms', 'Bathrooms', 'Age'}
    if required.issubset(df.columns):
        preds = []
        for _, row in df.iterrows():
            row_df = preprocess_input(
                row['Location'], row['Area (sqft)'], row['Bedrooms'],
                row['Bathrooms'], row['Age']
            )
            price = model.predict(row_df)[0]
            preds.append(round(price, 2))
        df["Predicted Price (₹ lakhs)"] = preds
        st.success("✅ Predictions Complete!")
        st.dataframe(df)
        st.download_button("⬇ Download Results", df.to_csv(index=False), file_name="predictions.csv")
    else:
        st.error(f"CSV must have columns: {', '.join(required)}")

# --------------------------
# 📎 Footer
# --------------------------
st.markdown("<div class='footer'>Made with ❤ by Harsh Oza</div>", unsafe_allow_html=True)
