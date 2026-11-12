import streamlit as st
import pandas as pd
import joblib

# This is a secondary prototype. Mental_Health.py is the primary app.
try:
    model = joblib.load("model.pkl")
    onehot_encoder = joblib.load("onehot_encoder.pkl")
except Exception:
    st.error("The saved model or encoder could not be loaded. Check compatible Python and scikit-learn versions.")
    st.stop()

label_mappings = {
    "sleep_Quality": {"Poor": 0, "Average": 1, "Good": 2},
    "stress_Level": {"Low": 0, "Medium": 1, "High": 2},
    "access_to_Mental_Health_Resource": {"No": 0, "Yes": 1},
    "physical_Activity": {"Not At All": 0, "Weekly": 1, "Daily": 2},
}

def label_encode(value, mapping):
    return mapping[value]

st.title("Employee Mental Health Model — Prototype")
st.warning("Educational demonstration only. This output is not a clinical assessment or diagnosis.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
industry = st.selectbox("Industry", ["Healthcare", "IT", "Education", "Finance", "Consulting", "Manufacturing"])
location = st.selectbox("Work Location", ["Onsite", "Remote", "Hybrid"])
sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])
stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
access_to_resources = st.selectbox("Access to Mental Health Resources", ["Yes", "No"])
physical_activity = st.selectbox("Physical Activity Level", ["Not At All", "Weekly", "Daily"])
region = st.selectbox("Region", ["Europe", "Asia", "North America", "South America", "Oceania", "Africa"])
hours_worked_per_week = st.number_input("Hours Worked Per Week", min_value=0, max_value=100, value=40)

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
    "region": [region],
})

try:
    encoded = onehot_encoder.transform(input_data[["gender", "industry", "Location", "region"]])
except Exception:
    st.error("The selected categories are not supported by the saved encoder. Retrain and package the model and encoder together.")
    st.stop()
encoded_df = pd.DataFrame(
    encoded,
    columns=onehot_encoder.get_feature_names_out(["gender", "industry", "Location", "region"]),
    index=input_data.index,
)
input_final = pd.concat(
    [input_data.drop(["gender", "industry", "Location", "region"], axis=1), encoded_df],
    axis=1,
)

if not hasattr(model, "feature_names_in_"):
    st.error("The saved model does not expose its expected feature names. Retrain/package the model and preprocessing together.")
    st.stop()

expected = list(model.feature_names_in_)
missing = [column for column in expected if column not in input_final.columns]
extra = [column for column in input_final.columns if column not in expected]
if missing:
    st.error("Input preprocessing does not match the saved model. Missing model features: " + ", ".join(missing))
    st.stop()
if extra:
    input_final = input_final.drop(columns=extra)
input_final = input_final[expected]

if st.button("Run experimental prediction"):
    try:
        prediction = model.predict(input_final)
        st.write("Model output class:", prediction[0])
    except Exception:
        st.error("Prediction failed. The saved model and input preprocessing may be incompatible.")
    st.caption("Interpret the class label using the dataset's documented target mapping; do not treat it as a diagnosis.")
