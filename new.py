import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("model.pkl")
  # Replace with your actual model file name
onehot_encoder = joblib.load("onehot_encoder.pkl")


# Load the scaler if used
scaler = joblib.load('scaler.pkl')  # Replace with the scaler file name if applicable

# Label encoding mappings (based on the categorical map you provided)
label_mappings = {
    "sleep_Quality": {"Poor": 0, "Average": 1, "Good": 2},
    "stress_Level": {"Low": 0, "Medium": 1, "High": 2},
    "access_to_Mental_Health_Resource": {"No": 0, "Yes": 1},
    "physical_Activity": {"Not At All": 0, "Weekly": 1, "Daily": 2}
}

# Function to apply label encoding based on the mappings
def label_encode(value, mapping):
    return mapping.get(value, -1)  # Return -1 for any unexpected input

# Define input fields for each feature
st.title("Employee Mental Health Prediction")

# Collecting inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
industry = st.selectbox("Industry", options=["Healthcare", "IT", "Education", "Finance", "Consulting", "Manufacturing"])
location = st.selectbox("Work Location", options=["Onsite", "Remote", "Hybrid"])
sleep_quality = st.selectbox("Sleep Quality", options=["Poor", "Average", "Good"])
stress_level = st.selectbox("Stress Level", options=["Low", "Medium", "High"])
access_to_resources = st.selectbox("Access to Mental Health Resources", options=["Yes", "No"])
physical_activity = st.selectbox("Physical Activity Level", options=["Not At All", "Weekly", "Daily"])
region = st.selectbox("Region", options=["Europe", "Asia", "North America", "South America", "Oceania", "Africa"])
hours_worked_per_week = st.number_input("Hours Worked Per Week", min_value=0, max_value=100, value=40)

# Prepare data for prediction
input_data = pd.DataFrame({
    "Age": [age],
    "Hours_Worked_Per_Week": [hours_worked_per_week],
    "sleep_Quality": [label_encode(sleep_quality, label_mappings["sleep_Quality"])],
    "stress_Level": [label_encode(stress_level, label_mappings["stress_Level"])],
    "access_to_Mental_Health_Resource": [label_encode(access_to_resources, label_mappings["access_to_Mental_Health_Resource"])],
    "physical_Activity": [label_encode(physical_activity, label_mappings["physical_Activity"])],
    "gender": [gender],
    "industry": [industry],
    "Location": [location],
    "region": [region]
})

#Apply one-hot encoding using the loaded encoder
onehot_encoded_data = onehot_encoder.transform(input_data[['gender', 'industry', 'Location', 'region']])

# Convert one-hot encoded data to DataFrame with appropriate column names
onehot_encoded_df = pd.DataFrame(onehot_encoded_data, columns=onehot_encoder.get_feature_names_out(['gender', 'industry', 'Location', 'region']))

# Concatenate the numerical features and the one-hot encoded categorical features
input_final = pd.concat([input_data.drop(['gender', 'industry', 'Location', 'region'], axis=1), onehot_encoded_df], axis=1)

# Ensure the input columns match the model’s expected features
missing_cols = set(model.feature_names_in_) - set(input_final.columns)
for col in missing_cols:
    input_final[col] = 0  # Add missing columns as 0

extra_cols = set(input_final.columns) - set(model.feature_names_in_)
input_final.drop(extra_cols, axis=1, inplace=True)

# Scale the input if scaler exists
#input_scaled = scaler.transform(input_final) if scaler else input_final

# Make prediction
if st.button("Predict Mental Health Condition"):
    prediction = model.predict(input_data)  # Adjust based on actual model expectations
    st.write("Predicted Mental Health Condition:", "Ill" if prediction[0] == 1 else "Fit")

