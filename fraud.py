import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Fraud Detection AI", layout="wide")
st.title("💳 AI Fraud Detection System")

# -------------------------
# LOAD DATA
# -------------------------
st.sidebar.header("📂 Upload Data")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    df = pd.read_csv("fraud_data.csv")  # default file

# -------------------------
# DATA PREPROCESSING
# -------------------------
le_location = LabelEncoder()
le_device = LabelEncoder()

df["location"] = le_location.fit_transform(df["location"])
df["device"] = le_device.fit_transform(df["device"])
X = df[["amount", "time", "location", "device"]]
y = df["fraud"]
# -------------------------
# TRAIN MODEL
# -------------------------
model = RandomForestClassifier()
model.fit(X, y)

# -------------------------
# SIDEBAR INPUT
# -------------------------
st.sidebar.header("🧾 Manual Transaction")

amount = st.sidebar.slider("Amount", 100, 5000, 1000)
time = st.sidebar.slider("Time (hour)", 0, 23, 12)
location = st.sidebar.selectbox("Location", ["Nagpur", "Mumbai", "Pune"])
device = st.sidebar.selectbox("Device", ["Mobile", "Laptop"])

location_enc = le_location.transform([location])[0]
device_enc = le_device.transform([device])[0]

input_data = np.array([[amount, time, location_enc, device_enc]])

# -------------------------
# PREDICTION
# -------------------------
prob = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]

st.subheader("🔍 Transaction Analysis")

col1, col2, col3 = st.columns(3)

col1.metric("💰 Amount", f"₹{amount}")
col2.metric("⏰ Time", f"{time}:00")
col3.metric("📍 Location", location)

# Risk level
if prob < 0.3:
    risk = "🟢 Low Risk"
elif prob < 0.7:
    risk = "🟡 Medium Risk"
else:
    risk = "🔴 High Risk"

st.markdown(f"### Risk Level: {risk}")

if prediction == 1:
    st.error(f"⚠️ Fraud Detected (Confidence: {prob:.2f})")
else:
    st.success(f"✅ Safe Transaction (Confidence: {prob:.2f})")
st.subheader("📊 Overview Dashboard")

total = len(df)
frauds = df["fraud"].sum()
normal = total - frauds
avg_amount = int(df["amount"].mean())

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", total)
c2.metric("Fraud Cases", frauds)
c3.metric("Normal", normal)
c4.metric("Avg Amount", f"₹{avg_amount}")
# -------------------------
# SIMPLE CHARTS (NO PLOTLY)
# -------------------------
st.subheader("🧠 Smart Insights")

high_amount = df[df["amount"] > 3000]["fraud"].mean()

if high_amount > 0.5:
    st.warning("⚠️ High-value transactions are more likely fraud")
else:
    st.info("💡 High-value transactions are mostly safe")

peak_time = df.groupby("time")["fraud"].mean().idxmax()

st.write(f"🚨 Most fraud happens around: {peak_time}:00 hours")

# -------------------------
# MAP SECTION
# -------------------------
st.subheader("🗺️ Fraud Map")

m = folium.Map(location=[20.5, 78.9], zoom_start=5)

for _, row in df.iterrows():
    color = "red" if row["fraud"] == 1 else "green"
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6,
        color=color,
        fill=True,
        popup=f"Fraud: {row['fraud']}"
    ).add_to(m)

st_folium(m, width=1000, height=400)

# -------------------------
# TABLE VIEW
# -------------------------
st.subheader("📋 Data Table")
st.dataframe(df)

# -------------------------
# DOWNLOAD
# -------------------------
st.subheader("📄 Download Data")

csv = df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="fraud_report.csv",
    mime="text/csv"
)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
