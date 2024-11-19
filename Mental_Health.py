
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")
onehot_encoder = joblib.load("onehot_encoder.pkl")

# Define label mappings for consistency
label_mappings = {
    "age": {"22-32": 0, "32-42": 1, ">42": 2},
    "sleep_Quality": {"Poor": 0, "Average": 1, "Good": 2},
    "stress_Level": {"Low": 0, "Medium": 1, "High": 2},
    "access_to_Mental_Health_Resource": {"No": 0, "Yes": 1},
    "physical_Activity": {"Not At All": 0, "Weekly": 1, "Daily": 2}
}


def label_encode(value, mapping):
    return mapping.get(value, -1)

# Title
st.title("Employee Mental Health Prediction")

# User inputs
#age = st.number_input("Age", min_value=18, max_value=65, value=30)
age = st.selectbox("Age", options=["22-32", "32-42", ">42"])
hours_worked_per_week = st.number_input("Hours Worked Per Week", min_value=0, max_value=100, value=40)
gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
industry = st.selectbox("Industry", options=["Healthcare", "IT", "Education", "Finance", "Consulting", "Manufacturing"])
location = st.selectbox("Work Location", options=["Onsite", "Remote", "Hybrid"])
region = st.selectbox("Region", options=["Europe", "Asia", "North America", "South America", "Oceania", "Africa"])

# Label encoded other inputs 
sleep_quality = label_encode(st.selectbox("Sleep Quality", ["Poor", "Average", "Good"]), label_mappings["sleep_Quality"])
stress_level = label_encode(st.selectbox("Stress Level", ["Low", "Medium", "High"]), label_mappings["stress_Level"])
access_to_mental_health_resource = label_encode(st.selectbox("Access to Mental Health Resource", ["No", "Yes"]), label_mappings["access_to_Mental_Health_Resource"])
physical_activity = label_encode(st.selectbox("Physical Activity", ["Not At All", "Weekly", "Daily"]), label_mappings["physical_Activity"])

# Convert inputs into a DataFrame for processing
input_data = pd.DataFrame([[age, hours_worked_per_week, gender, industry, location, region, sleep_quality, stress_level, access_to_mental_health_resource, physical_activity]],
                          columns=["age", "hours_worked_per_week", "gender", "industry", "Location", "region", "sleep_Quality", "stress_Level", "access_to_Mental_Health_Resource", "physical_Activity"])

encoded_input = onehot_encoder.transform(input_data[["gender", "industry", "Location", "region"]])
encoded_input_df = pd.DataFrame(encoded_input, columns=onehot_encoder.get_feature_names_out(["gender", "industry", "Location", "region"]))

input_data = input_data.drop(["gender", "industry", "Location", "region"], axis=1)
input_data = pd.concat([input_data, encoded_input_df], axis=1)

missing_cols = set(model.feature_names_in_) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0  

input_data = input_data[model.feature_names_in_]  


if st.button("Predict Mental Health Condition"):
    prediction = model.predict(input_data)  
    st.write("Predicted Mental Health Condition:", "Ill" if prediction[0] == 1 else "Fit")




