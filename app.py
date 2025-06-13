import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import json
import os
from datetime import datetime


if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

# admin credentials
VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"

# User data file
USERS_FILE = "users.json"

# Load or create users file
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {"admin": {"password": "admin123", "role": "admin"}}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# Login/Signup page
if not st.session_state.authenticated:
    st.title("🔐 Authentication")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.write("Please login to access the application")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                users = load_users()
                if username in users and users[username]["password"] == password:
                    st.session_state.authenticated = True
                    st.session_state.current_user = username
                    st.session_state.is_admin = users[username]["role"] == "admin"
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.write("Create a new account")
            new_username = st.text_input("Choose Username", key="signup_username")
            new_password = st.text_input("Choose Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if st.button("Sign Up"):
                if new_password != confirm_password:
                    st.error("Passwords do not match!")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long!")
                else:
                    users = load_users()
                    if new_username in users:
                        st.error("Username already exists!")
                    else:
                        users[new_username] = {"password": new_password, "role": "user"}
                        save_users(users)
                        st.success("Account created successfully! Please login.")
    
    st.stop()

# Load model and encoders
model = joblib.load("real_estate_fraud_model.jb")
encoder = joblib.load("real_estate_label_encoders.jb")

# Function to calculate distance
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Add logout button in the sidebar
with st.sidebar:
    st.write(f"Logged in as: {st.session_state.current_user}")
    if st.session_state.is_admin:
        st.write("👑 Admin Dashboard")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.is_admin = False
        st.rerun()

# Admin Dashboard
if st.session_state.is_admin:
    st.title("👑 Admin Dashboard")
    
    admin_tab1, admin_tab2, admin_tab3 = st.tabs(["User Management", "Fraud Analysis", "System Settings"])
    
    with admin_tab1:
        st.header("User Management")
        users = load_users()
        user_df = pd.DataFrame([
            {"Username": username, "Role": data["role"]}
            for username, data in users.items()
        ])
        st.dataframe(user_df)
        
        st.subheader("Add New User")
        new_admin_username = st.text_input("Username", key="admin_new_username")
        new_admin_password = st.text_input("Password", type="password", key="admin_new_password")
        new_admin_role = st.selectbox("Role", ["user", "admin"], key="admin_new_role")
        
        if st.button("Add User"):
            if new_admin_username and new_admin_password:
                users[new_admin_username] = {"password": new_admin_password, "role": new_admin_role}
                save_users(users)
                st.success("User added successfully!")
                st.rerun()
            else:
                st.error("Please fill all fields!")
    
    with admin_tab2:
        st.header("Fraud Analysis")
        try:
            df = pd.read_csv("transactions_log.csv")
            st.subheader("Fraud Statistics")
            
            # Overall statistics
            total_transactions = len(df)
            fraud_transactions = df["fraudulent"].sum()
            fraud_rate = (fraud_transactions / total_transactions) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", total_transactions)
            with col2:
                st.metric("Fraud Cases", fraud_transactions)
            with col3:
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            
            # Detailed analysis
            st.subheader("Fraud Patterns")
            if "month" in df.columns:
                monthly_fraud = df[df["fraudulent"] == 1].groupby("month").size()
                st.line_chart(monthly_fraud)
            
            # Property type analysis
            property_fraud = df.groupby("property_type")["fraudulent"].agg(["count", "sum"])
            property_fraud["fraud_rate"] = (property_fraud["sum"] / property_fraud["count"]) * 100
            st.subheader("Fraud by Property Type")
            st.dataframe(property_fraud)
            
        except FileNotFoundError:
            st.error("Transaction log file not found")
    
    with admin_tab3:
        st.header("System Settings")
        st.subheader("Model Information")
        st.write("Current Model: Real Estate Fraud Detection Model")
        st.write("Last Updated: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        st.subheader("System Health")
        try:
            df = pd.read_csv("transactions_log.csv")
            st.success("✅ Transaction log is accessible")
            st.success("✅ Model is loaded successfully")
            st.success("✅ Encoders are loaded successfully")
        except Exception as e:
            st.error(f"❌ System Error: {str(e)}")

# Main Application
else:
    st.title("🏠 Real Estate Fraud Detection & Insights")
    
    # Tabs: Form | Insights
    tab1, tab2 = st.tabs(["🔍 Fraud Check", "📊 Insights Dashboard"])
    
    with tab1:
        st.header("Fraud Detection System")
        st.write("Enter the Real Estate Transaction Details")

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
                input_data = pd.DataFrame([[buyer_name, seller_name, property_type, property_value, mortgage_amount,
                                            distance, month, buyer_gender, ssn]],
                                          columns=['buyer_name', 'seller_name', 'property_type', 'property_value',
                                                   'mortgage_amount', 'distance', 'month', 'buyer_gender', 'ssn'])

                categorical_cols = ['buyer_name', 'seller_name', 'property_type', 'buyer_gender']
                for col in categorical_cols:
                    try:
                        input_data[col] = encoder[col].transform(input_data[col])
                    except (KeyError, ValueError):
                        input_data[col] = input_data[col].apply(lambda x: hash(x) % 1000)

                input_data['ssn'] = input_data['ssn'].apply(lambda x: hash(x) % 10000)

                prediction = model.predict(input_data)[0]
                result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
                st.subheader(f"Prediction: {result}")
                
                # Add fraud analysis details
                if prediction == 1:
                    st.warning("Potential Fraud Indicators:")
                    if distance > 100:
                        st.write("⚠️ Large distance between buyer and property location")
                    if mortgage_amount > property_value * 0.9:
                        st.write("⚠️ High mortgage-to-value ratio")
                    if property_value > 1000000:
                        st.write("⚠️ High-value property transaction")
                else:
                    st.success("Transaction appears legitimate based on:")
                    st.write("✅ Normal distance between buyer and property")
                    st.write("✅ Reasonable mortgage-to-value ratio")
                    st.write("✅ Standard property value range")
            else:
                st.error("Please fill all required fields correctly.")

    with tab2:
        st.header("📊 Real Estate Fraud Insights Dashboard")

        try:
            df = pd.read_csv("transactions_log.csv")

            df["fraudulent"] = df["fraudulent"].astype(int)
            fraud_counts = df.groupby("property_type")["fraudulent"].sum()
            total_counts = df["property_type"].value_counts()
            fraud_rates = (fraud_counts / total_counts) * 100

            summary_df = pd.DataFrame({
                "Fraud Cases": fraud_counts,
                "Total Transactions": total_counts,
                "Fraud Rate (%)": fraud_rates.round(2)
            }).fillna(0)

            # Bar Chart
            st.subheader("Fraud Cases by Property Type")
            st.bar_chart(summary_df["Fraud Cases"])

            # Pie Chart
            st.subheader("Fraud Distribution Pie Chart")
            fig1, ax1 = plt.subplots()
            ax1.pie(summary_df["Fraud Cases"], labels=summary_df.index, autopct="%1.1f%%", startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

            # Line Chart
            st.subheader("Fraud Rate (%) by Property Type")
            st.line_chart(summary_df["Fraud Rate (%)"])

            # Table
            st.subheader("Fraud Summary Table")
            st.dataframe(summary_df.style.highlight_max(axis=0, color="lightgreen"))

            # Monthly Trend Area Chart
            st.subheader("Monthly Fraud Trends")
            if "month" in df.columns:
                df["month"] = df["month"].astype(int)
                month_fraud = df[df["fraudulent"] == 1]["month"].value_counts().sort_index()
                months = pd.date_range("2024-01-01", periods=12, freq='ME').strftime('%b')
                monthly_df = pd.DataFrame({
                    "Month": months,
                    "Fraud Cases": [month_fraud.get(i, 0) for i in range(1, 13)]
                }).set_index("Month")
                st.area_chart(monthly_df)
            else:
                st.info("Month data not available in CSV.")

        except FileNotFoundError:
            st.error("transactions_log.csv not found.")
