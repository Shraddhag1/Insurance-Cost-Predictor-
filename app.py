import streamlit as st
import pandas as pd
import joblib

model = joblib.load("gradient_boost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Medical Insurance Cost Predictor")

age = st.slider("Age", 18, 100, 25)
bmi = st.slider("BMI", 15.0, 40.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker?", ["yes", "no"])
region = st.selectbox("Region", ["Mumbai", "Delhi", "Pune", "Bangalore"])

input_data = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "children": children,
    "region_Delhi": 1 if region == "Delhi" else 0,
    "region_Mumbai": 1 if region == "Mumbai" else 0,
    "region_Pune": 1 if region == "Pune" else 0,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0
}])


input_data = input_data[['age', 'bmi', 'children', 
                         'region_Delhi', 'region_Mumbai', 'region_Pune', 
                         'sex_male', 'smoker_yes']]

scaled_input = scaler.transform(input_data)
predicted_charge = model.predict(scaled_input)

st.subheader(f"Predicted Insurance Cost: ₹{predicted_charge[0]:,.2f}")
