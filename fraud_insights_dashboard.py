import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame({
    "Property Type": ["Residential", "Commercial", "Industrial", "Land"],
    "Fraud Cases": [25, 18, 10, 7],
    "Total Transactions": [120, 80, 40, 35]
})

data["Fraud Rate (%)"] = (data["Fraud Cases"] / data["Total Transactions"]) * 100

st.title("Real Estate Fraud Insights Dashboard")

st.subheader("Fraud Cases by Property Type")
st.bar_chart(data.set_index("Property Type")["Fraud Cases"])

st.subheader("Fraud Distribution Pie Chart")
fig1, ax1 = plt.subplots()
ax1.pie(data["Fraud Cases"], labels=data["Property Type"], autopct="%1.1f%%", startangle=90,
        colors=["#FF6F61", "#6A5ACD", "#20B2AA", "#FFB347"])
ax1.axis("equal")
st.pyplot(fig1)

st.subheader("Fraud Rate (%) by Property Type")
st.line_chart(data.set_index("Property Type")["Fraud Rate (%)"])

st.subheader("Fraud Summary Table")
st.dataframe(data.style.highlight_max(axis=0, color='lightgreen'))

st.subheader("Monthly Fraud Trends (Simulated)")
months = pd.date_range(start="2024-01-01", periods=12, freq='ME').strftime('%b')
monthly_data = pd.DataFrame({
    "Month": months,
    "Fraud Cases": np.random.randint(5, 30, 12)
}).set_index("Month")
st.area_chart(monthly_data)

st.markdown("---")
st.markdown("🔍 _This dashboard shows simulated insights.")

# streamlit run fraud_insights_dashboard.py

