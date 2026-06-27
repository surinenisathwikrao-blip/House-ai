import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="House AI Pro", page_icon="🏠", layout="centered")

st.title("🏠 Telangana Construction AI Pro")
st.markdown("Professional construction cost estimation system")

# Load data
data = pd.read_csv("house_data.csv")

X = data[["District", "Area", "Type", "Floors"]]
y = data["Cost"]

# Model
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["District", "Type"]),
    ("num", "passthrough", ["Area", "Floors"])
])

model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=150, random_state=42))
])

model.fit(X, y)

# Inputs
district = st.selectbox("District", data["District"].unique())
area = st.slider("Area (sq.ft)", 500, 5000, 1200)
house_type = st.selectbox("House Type", ["Budget", "Middle", "Luxury"])
floors = st.number_input("Floors", 1, 10, 1)

if st.button("Generate Report"):

    input_df = pd.DataFrame({
        "District": [district],
        "Area": [area],
        "Type": [house_type],
        "Floors": [floors]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"Total Estimated Cost: ₹ {int(prediction):,}")

    # Breakdown
    labels = ["Cement", "Steel", "Bricks", "Labor", "Finishing"]
    values = [
        prediction * 0.20,
        prediction * 0.25,
        prediction * 0.10,
        prediction * 0.30,
        prediction * 0.15
    ]

    st.subheader("📊 Cost Breakdown")

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.subheader("📄 Report Summary")
    st.write({
        "District": district,
        "Area": area,
        "Type": house_type,
        "Floors": floors,
        "Total Cost": int(prediction)
    })

    st.info("This is an estimated value for planning purposes only.")
