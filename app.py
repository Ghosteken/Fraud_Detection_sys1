import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
from geopy.distance import geodesic

model = joblib.load("real_estate_fraud_model.jb")
encoder = joblib.load("real_estate_label_encoders.jb")

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

st.title("Real Estate Fraud Detection System")
st.write("Enter the Real Estate Transaction Details Below")

buyer_name = st.text_input("Buyer Name")
seller_name = st.text_input("Seller Name")
property_type = st.selectbox("Property Type", ["Residential", "Commercial", "Industrial", "Land"])
property_value = st.number_input("Property Value (USD)", min_value=1000.0, format="%.2f")
mortgage_amount = st.number_input("Mortgage Amount (USD)", min_value=0.0, format="%.2f")

location_lat = st.number_input("Property Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
location_long = st.number_input("Property Longitude", min_value=-180.0, max_value=180.0, format="%.6f")
buyer_lat = st.number_input("Buyer Address Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
buyer_long = st.number_input("Buyer Address Longitude", min_value=-180.0, max_value=180.0, format="%.6f")

month = st.slider("Transaction Month", 1, 12, 6)
buyer_gender = st.selectbox("Buyer Gender", ["Male", "Female"])
ssn = st.text_input("Buyer's SSN (last 4 digits)")

distance = None
if all(-90 <= lat <= 90 for lat in [location_lat, buyer_lat]) and \
   all(-180 <= lon <= 180 for lon in [location_long, buyer_long]):
    distance = haversine(location_lat, location_long, buyer_lat, buyer_long)
else:
    st.warning("Invalid latitude or longitude values.")

if st.button("Check for Fraud"):
    if buyer_name and seller_name and ssn and distance is not None:
        input_data = pd.DataFrame([[
            buyer_name, seller_name, property_type, property_value, mortgage_amount,
            distance, month, buyer_gender, ssn
        ]], columns=[
            'buyer_name', 'seller_name', 'property_type', 'property_value',
            'mortgage_amount', 'distance', 'month', 'buyer_gender', 'ssn'
        ])

        categorical_cols = ['buyer_name', 'seller_name', 'property_type', 'buyer_gender']
        for col in categorical_cols:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except (KeyError, ValueError):
                input_data[col] = input_data[col].apply(lambda x: hash(x) % 1000)

        input_data['ssn'] = input_data['ssn'].apply(lambda x: hash(x) % 10000)

        prediction = model.predict(input_data)[0]
        result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
        st.subheader(f"Prediction: {result}")
    else:
        st.error("Please fill all required fields correctly.")


# streamlit run app.py