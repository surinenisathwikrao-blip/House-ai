import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")

st.set_page_config(page_title="Telangana House AI", page_icon="🏠")

st.title("🏠 Telangana Construction AI Pro")

st.markdown("Get full construction estimate + material breakdown")

district = st.selectbox("District", [
    "Hyderabad", "Warangal", "Karimnagar",
    "Nizamabad", "Khammam", "Vijayawada"
])

area = st.number_input("Area (sq.ft)", 500, 10000, 1200)

house_type = st.selectbox("House Type", ["Budget", "Middle", "Luxury"])

floors = st.number_input("Floors", 1, 10, 1)

if st.button("Generate Estimate"):
    
    input_df = pd.DataFrame({
        "District": [district],
        "Area": [area],
        "Type": [house_type],
        "Floors": [floors]
    })

    base_cost = model.predict(input_df)[0]

    # Material breakdown (approx logic)
    cement_cost = base_cost * 0.20
    steel_cost = base_cost * 0.25
    bricks_cost = base_cost * 0.10
    labor_cost = base_cost * 0.30
    finishing_cost = base_cost * 0.15

    # Time estimation
    if area < 1200:
        months = 6
    elif area < 2000:
        months = 9
    else:
        months = 12

    st.success(f"Total Estimated Cost: ₹ {int(base_cost):,}")

    st.subheader("📦 Material Breakdown")
    st.write(f"Cement: ₹ {int(cement_cost):,}")
    st.write(f"Steel: ₹ {int(steel_cost):,}")
    st.write(f"Bricks: ₹ {int(bricks_cost):,}")
    st.write(f"Labor: ₹ {int(labor_cost):,}")
    st.write(f"Finishing: ₹ {int(finishing_cost):,}")

    st.subheader("⏱️ Construction Time")
    st.info(f"Estimated Time: {months} months")