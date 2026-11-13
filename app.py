"""Streamlit interface for the reproducibly trained demonstration pipeline."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "mental_health_pipeline.joblib"

st.set_page_config(page_title="Employee Mental Health Model", page_icon="📊")
st.title("Employee Mental Health — Experimental Demonstration")
st.warning(
    "Educational demonstration only. This model is not a clinical assessment, "
    "screening instrument or diagnosis. Do not use predictions for hiring, "
    "employee evaluation, treatment or other consequential decisions."
)

if not MODEL_PATH.exists():
    st.error("Model artifact not found. Run `python train.py` before starting the app.")
    st.stop()

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

with st.form("prediction_form"):
    age = st.selectbox("Age group", ["22-32", "32-42", ">42"])
    hours = st.number_input("Hours worked per week", min_value=0, max_value=100, value=40)
    access = st.selectbox("Access to mental-health resources", ["No", "Yes"])
    sleep = st.selectbox("Sleep quality", ["Poor", "Average", "Good"])
    stress = st.selectbox("Stress level", ["Low", "Medium", "High"])
    activity = st.selectbox("Physical activity", ["Not At All", "Weekly", "Daily"])
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    industry = st.selectbox(
        "Industry",
        ["Healthcare", "IT", "Education", "Finance", "Consulting", "Manufacturing", "Retail"],
    )
    location = st.selectbox("Work arrangement", ["Onsite", "Remote", "Hybrid"])
    region = st.selectbox(
        "Region",
        ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"],
    )
    submitted = st.form_submit_button("Run experimental prediction")

if submitted:
    row = pd.DataFrame(
        [{
            "Age": age,
            "Hours_Worked_Per_Week": hours,
            "access_to_Mental_Health_Resource": access,
            "sleep_Quality": sleep,
            "stress_Level": stress,
            "physical_Activity": activity,
            "gender": gender,
            "industry": industry,
            "Location": location,
            "region": region,
        }]
    )
    try:
        prediction = pipeline.predict(row)[0]
        st.write("Model output class:", prediction)
        st.caption(
            "The output reproduces the dataset label only; it is not a diagnosis "
            "or an assessment of an individual's mental health."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
