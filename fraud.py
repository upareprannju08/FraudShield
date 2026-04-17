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
le = LabelEncoder()
df["location"] = le.fit_transform(df["location"])
df["device"] = le.fit_transform(df["device"])

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

location_enc = le.transform([location])[0]
device_enc = le.transform([device])[0]

input_data = np.array([[amount, time, location_enc, device_enc]])

# -------------------------
# PREDICTION
# -------------------------
prob = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]

st.subheader("🔍 Prediction Result")

if prediction == 1:
    st.error(f"⚠️ Fraud Detected! Risk Score: {prob:.2f}")
else:
    st.success(f"✅ Safe Transaction. Risk Score: {prob:.2f}")

# -------------------------
# SIMPLE CHARTS (NO PLOTLY)
# -------------------------
st.subheader("📊 Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("### Transaction Amount Distribution")
    st.bar_chart(df["amount"])

with col2:
    st.write("### Fraud Count")
    fraud_counts = df["fraud"].value_counts()
    st.bar_chart(fraud_counts)

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
