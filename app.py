# ==========================================================
# Imports
# ==========================================================

import streamlit as st
import joblib

# ==========================================================
# Load Trained Pipeline
# ==========================================================

pipeline = joblib.load(
    "model/stress_detection_pipeline.pkl"
)

# ==========================================================
# Streamlit Page Configuration
# ==========================================================


st.set_page_config(
    page_title="Reddit Mental Health Classifier",
     page_icon="🧠",
    layout="centered"
)

st.title("🧠 Reddit Mental Health Stress Detector")

st.write(
    "Enter a Reddit post below and the model will predict "
    "whether it indicates signs of stress."
)

user_input = st.text_area(
    "Reddit Post",
    height=200
)

if st.button("Predict"):

    if user_input.strip():

        prediction = pipeline.predict(
            [user_input]
        )

        probabilities = pipeline.predict_proba(
            [user_input]
        )

        stress_probability = probabilities[0][1]
        no_stress_probability = probabilities[0][0]

        prediction_label = (
            "Stress"
            if prediction[0] == 1
            else "No Stress"
        )

        st.subheader("Prediction")

        if prediction[0] == 1:
            st.error("🚨 Stress Detected")
        else:
            st.success("✅ No Stress Detected")
        
        st.metric(
            "Confidence",
            f"{max(probabilities[0])*100:.2f}%"
        )

        st.progress(stress_probability)

        st.write(
            f"**Stress Probability:** {stress_probability * 100:.2f}%"
        )

        st.write(
            f"**No Stress Probability:** {no_stress_probability * 100:.2f}%"
        )