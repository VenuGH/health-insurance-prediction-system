import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "..", "notebooks")
scaler = joblib.load(os.path.join(NOTEBOOKS_DIR, "scaler.pkl"))
le_gender = joblib.load(os.path.join(NOTEBOOKS_DIR, "label_encoder_gender.pkl"))
le_diabetic = joblib.load(os.path.join(NOTEBOOKS_DIR, "label_encoder_diabetic.pkl"))
le_smoker = joblib.load(os.path.join(NOTEBOOKS_DIR, "label_encoder_smoker.pkl"))
model = joblib.load(os.path.join(NOTEBOOKS_DIR, "best_model.pkl"))

st.set_page_config(page_title = "Insurance Claim Predictor", layout="centered")
st.title("Health Insurance Payment Prediction")
st.write("Enter the details below to estimate your insurance payment amount.")

st.markdown(
    """
    <style>
    /* === Main app background (your image + darker overlay for contrast) === */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(
            rgba(10, 74, 110, 0.78),   /* dark teal-cyan tint – adjust 0.65–0.85 */
            rgba(10, 74, 110, 0.78)
        ),
        url("https://static.vecteezy.com/system/resources/previews/010/568/309/original/health-care-icon-pattern-medical-innovation-concept-background-design-vector.jpg");
        
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }

    /* === Make title & subtitle much more readable === */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {
        color: #ffffff !important;           /* pure white */
        text-shadow: 0 3px 8px rgba(0,0,0,0.9),    /* strong dark shadow */
                    0 0 12px rgba(0,180,255,0.4);  /* subtle cyan glow */
        letter-spacing: 0.5px;
    }

    /* Instructions text / small text */
    p, .stMarkdown p, label {
        color: #e0f7ff !important;           /* very light cyan-white */
        text-shadow: 0 1px 4px rgba(0,0,0,0.8);
    }

    /* === Input fields & widgets clearer === */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.88) !important;
        color: #0a1f44 !important;                    /* dark blue text */
        border: 1px solid #4da8da !important;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }

    /* Labels above inputs */
    .stNumberInput label, .stSelectbox label, .stRadio label {
        color: #b3e0ff !important;
        font-weight: 600;
        text-shadow: 0 1px 3px rgba(0,0,0,0.7);
    }

    /* Buttons (predict button etc.) */
    .stButton > button {
        background-color: #00aaff !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,170,255,0.4);
    }
    .stButton > button:hover {
        background-color: #0088cc !important;
    }

    /* Optional: Frosted effect on main content block for cleaner look */
    .main .block-container {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.18);
        padding: 2rem 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with st.form("input_form"):
    col1,col2 = st.columns(2)
    with col1:
        age = st.number_input("Age",min_value=0, max_value=100, value = 30)
        bmi = st.number_input("BMI",min_value=10, max_value=60, value = 20)
        children = st.number_input("Number of Children", min_value=0, max_value=8, value=0)
    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
        gender = st.selectbox("Gender", options=le_gender.classes_)
        diabetic = st.selectbox("Diabetic", options=le_diabetic.classes_)
        smoker =st.selectbox("Smoker", options=le_smoker.classes_)


    submitted = st.form_submit_button("Predict Payment")

if submitted:

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bloodpressure": [bloodpressure],
        "diabetic": [diabetic],
        "children": [children],
        "smoker": [smoker],
         "bmi" : [bmi],
    })


    input_data["gender"] = le_gender.transform(input_data["gender"])
    input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
    input_data["smoker"] = le_smoker.transform(input_data["smoker"])
    
    num_cols = ["age","bmi","bloodpressure","children"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    prediction = model.predict(input_data)[0]
    st.success(f"**Estimated Insurance Payment Amount:** ${prediction:,.2f}")

