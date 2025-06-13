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
import shutil
from pathlib import Path


if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = {}

# admin credentials
VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"

# File paths
USERS_FILE = "users.json"
FRAUD_METRICS_FILE = "fraud_metrics.json"
CHECKS_HISTORY_FILE = "fraud_checks_history.json"
UPLOADS_DIR = "document_uploads"

# Create uploads directory if it doesn't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Load fraud metrics
def load_fraud_metrics():
    with open(FRAUD_METRICS_FILE, 'r') as f:
        return json.load(f)

# Save check history
def save_check_history(check_data):
    if os.path.exists(CHECKS_HISTORY_FILE):
        with open(CHECKS_HISTORY_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(check_data)
    
    with open(CHECKS_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

# Document handling functions
def save_uploaded_file(uploaded_file, transaction_id, document_type):
    """Save uploaded document with proper naming and organization"""
    if uploaded_file is not None:
        # Create transaction directory
        transaction_dir = os.path.join(UPLOADS_DIR, transaction_id)
        os.makedirs(transaction_dir, exist_ok=True)
        
        # Save file with proper extension
        file_extension = Path(uploaded_file.name).suffix
        file_path = os.path.join(transaction_dir, f"{document_type}{file_extension}")
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    return None

def verify_document(file_path):
    """Basic document verification (can be enhanced with actual document verification logic)"""
    if file_path and os.path.exists(file_path):
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Basic checks
        is_valid = True
        issues = []
        
        # Check file size (max 10MB)
        if file_size > 10 * 1024 * 1024:
            is_valid = False
            issues.append("File size exceeds 10MB limit")
        
        # Check file extension
        allowed_extensions = ['.pdf', '.jpg', '.jpeg', '.png']
        if not any(file_path.lower().endswith(ext) for ext in allowed_extensions):
            is_valid = False
            issues.append("Invalid file format. Allowed formats: PDF, JPG, JPEG, PNG")
        
        return {
            "is_valid": is_valid,
            "issues": issues,
            "file_size": file_size,
            "file_path": file_path
        }
    return {
        "is_valid": False,
        "issues": ["No file uploaded"],
        "file_size": 0,
        "file_path": None
    }

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
    
    admin_tab1, admin_tab2, admin_tab3, admin_tab4 = st.tabs(["User Management", "Fraud Analysis", "System Settings", "Fraud Metrics"])
    
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
    
    with admin_tab4:
        st.header("Fraud Detection Metrics")
        metrics = load_fraud_metrics()
        
        for metric_id, metric in metrics["metrics"].items():
            with st.expander(f"{metric['name']} ({metric['risk_level'].upper()})"):
                st.write("**Description:**", metric["description"])
                st.write("**Nigerian Context:**", metric["nigerian_context"])
                if "threshold" in metric:
                    st.write("**Threshold:**", f"{metric['threshold']} {metric['unit']}")
                if "required_documents" in metric:
                    st.write("**Required Documents:**")
                    for doc in metric["required_documents"]:
                        st.write(f"- {doc}")
                if "high_risk_areas" in metric:
                    st.write("**High Risk Areas:**")
                    for area in metric["high_risk_areas"]:
                        st.write(f"- {area}")

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
        property_value = st.number_input("Property Value (NGN)", min_value=1000000.0, format="%.2f")
        mortgage_amount = st.number_input("Mortgage Amount (NGN)", min_value=0.0, format="%.2f")
        property_size = st.number_input("Property Size (sqm)", min_value=1.0, format="%.2f")

        location_lat = st.number_input("Property Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
        location_long = st.number_input("Property Longitude", min_value=-180.0, max_value=180.0, format="%.6f")
        buyer_lat = st.number_input("Buyer Address Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
        buyer_long = st.number_input("Buyer Address Longitude", min_value=-180.0, max_value=180.0, format="%.6f")

        transaction_days = st.number_input("Transaction Processing Time (days)", min_value=1, value=45)
        buyer_gender = st.selectbox("Buyer Gender", ["Male", "Female"])
        ssn = st.text_input("Buyer's SSN (last 4 digits)")

        # Document upload section
        st.subheader("Required Documents")
        metrics = load_fraud_metrics()
        required_docs = metrics["metrics"]["document_verification_check"]["required_documents"]
        
        # Document selection
        st.write("Select documents to upload:")
        selected_docs = st.multiselect(
            "Choose documents",
            options=required_docs,
            default=[],
            help="Select one or more documents to upload"
        )
        
        # Document upload interface
        uploaded_files = {}
        document_status = {}
        
        if selected_docs:
            st.write("---")
            st.write("Upload selected documents:")
            
            # Create columns for document upload
            cols = st.columns(2)
            for idx, doc in enumerate(selected_docs):
                with cols[idx % 2]:
                    st.write(f"**{doc}**")
                    uploaded_file = st.file_uploader(
                        f"Upload {doc}",
                        type=['pdf', 'jpg', 'jpeg', 'png'],
                        key=f"upload_{doc}"
                    )
                    
                    if uploaded_file is not None:
                        # Generate unique transaction ID
                        transaction_id = f"{buyer_name}_{seller_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # Save and verify document
                        file_path = save_uploaded_file(uploaded_file, transaction_id, doc)
                        verification_result = verify_document(file_path)
                        
                        uploaded_files[doc] = file_path
                        document_status[doc] = verification_result
                        
                        if verification_result["is_valid"]:
                            st.success("✅ Document uploaded and verified successfully")
                        else:
                            st.error("❌ Document verification failed:")
                            for issue in verification_result["issues"]:
                                st.write(f"- {issue}")
        
        # Show upload status
        if uploaded_files:
            st.write("---")
            st.subheader("Upload Status")
            for doc, status in document_status.items():
                if status["is_valid"]:
                    st.success(f"✅ {doc}: Uploaded and verified")
                else:
                    st.error(f"❌ {doc}: Failed verification")
                    for issue in status["issues"]:
                        st.write(f"- {issue}")

        distance = None
        if all(-90 <= lat <= 90 for lat in [location_lat, buyer_lat]) and \
           all(-180 <= lon <= 180 for lon in [location_long, buyer_long]):
            distance = haversine(location_lat, location_long, buyer_lat, buyer_long)
        else:
            st.warning("Invalid latitude or longitude values.")

        if st.button("Check for Fraud"):
            if buyer_name and seller_name and ssn and distance is not None:
                # Load fraud metrics
                metrics = load_fraud_metrics()
                
                # Perform checks
                checks = {
                    "timestamp": datetime.now().isoformat(),
                    "buyer_name": buyer_name,
                    "seller_name": seller_name,
                    "property_type": property_type,
                    "checks": {},
                    "documents": {}
                }
                
                # Document verification
                all_docs_valid = True
                for doc in required_docs:
                    if doc in document_status:
                        status = document_status[doc]
                        checks["documents"][doc] = {
                            "is_valid": status["is_valid"],
                            "issues": status["issues"],
                            "file_path": status["file_path"]
                        }
                        if not status["is_valid"]:
                            all_docs_valid = False
                    else:
                        checks["documents"][doc] = {
                            "is_valid": False,
                            "issues": ["Document not uploaded"],
                            "file_path": None
                        }
                        all_docs_valid = False
                
                # Distance check
                checks["checks"]["distance"] = {
                    "value": distance,
                    "threshold": metrics["metrics"]["distance_check"]["threshold"],
                    "passed": distance <= metrics["metrics"]["distance_check"]["threshold"]
                }
                
                # Property value check
                checks["checks"]["property_value"] = {
                    "value": property_value,
                    "threshold": metrics["metrics"]["property_value_check"]["threshold"],
                    "passed": property_value <= metrics["metrics"]["property_value_check"]["threshold"]
                }
                
                # Mortgage ratio check
                mortgage_ratio = mortgage_amount / property_value if property_value > 0 else 0
                checks["checks"]["mortgage_ratio"] = {
                    "value": mortgage_ratio,
                    "threshold": metrics["metrics"]["mortgage_ratio_check"]["threshold"],
                    "passed": mortgage_ratio <= metrics["metrics"]["mortgage_ratio_check"]["threshold"]
                }
                
                # Transaction timing check
                checks["checks"]["transaction_timing"] = {
                    "value": transaction_days,
                    "threshold": metrics["metrics"]["transaction_timing_check"]["threshold"],
                    "passed": transaction_days >= metrics["metrics"]["transaction_timing_check"]["threshold"]
                }
                
                # Price per sqm check
                price_per_sqm = property_value / property_size if property_size > 0 else 0
                checks["checks"]["price_per_sqm"] = {
                    "value": price_per_sqm,
                    "threshold": metrics["metrics"]["price_per_sqm_check"]["threshold"],
                    "passed": price_per_sqm <= metrics["metrics"]["price_per_sqm_check"]["threshold"]
                }
                
                # Document verification check
                checks["checks"]["document_verification"] = {
                    "value": all_docs_valid,
                    "passed": all_docs_valid
                }
                
                # Save checks to history
                save_check_history(checks)
                
                # Calculate risk score
                risk_score = 0
                for check in checks["checks"].values():
                    if not check["passed"]:
                        metric_id = next(k for k, v in metrics["metrics"].items() 
                                      if v["threshold"] == check.get("threshold"))
                        risk_score += metrics["risk_levels"][metrics["metrics"][metric_id]["risk_level"]]["weight"]
                
                # Display results
                st.subheader("Fraud Detection Results")
                
                if risk_score >= 5:
                    st.error("🚨 High Risk Transaction")
                elif risk_score >= 3:
                    st.warning("⚠️ Medium Risk Transaction")
                else:
                    st.success("✅ Low Risk Transaction")
                
                st.subheader("Detailed Check Results")
                for check_name, check_result in checks["checks"].items():
                    metric = next(m for m in metrics["metrics"].values() if m["name"].lower().replace(" ", "_") == check_name)
                    if not check_result["passed"]:
                        st.error(f"❌ {metric['name']}")
                        st.write(f"Reason: {metric['description']}")
                        st.write(f"Nigerian Context: {metric['nigerian_context']}")
                    else:
                        st.success(f"✅ {metric['name']}")
                
                # Display document verification results
                st.subheader("Document Verification Results")
                for doc, status in checks["documents"].items():
                    if status["is_valid"]:
                        st.success(f"✅ {doc}: Verified")
                    else:
                        st.error(f"❌ {doc}: Failed")
                        for issue in status["issues"]:
                            st.write(f"- {issue}")
                
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
