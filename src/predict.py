# ==============================================================
# Imports
# ==============================================================

import joblib

# ===============================================================
# Load Trained Pipeline 
# ===============================================================

pipeline = joblib.load(
    "model/stress_detection_pipeline.pkl"
)

# ==========================================================
# Sample Reddit Post
# ==========================================================

sample_post = """
I have been feeling extremely anxious lately.
I cannot sleep.
I constantly worry about my future.
Nothing feels enjoyable anymore.
"""

# ==========================================================
# Predict Stress Level
# ==========================================================

prediction = pipeline.predict(
    [sample_post]
)

probabilities = pipeline.predict_proba(
    [sample_post]
)

stress_probability = probabilities[0][1]
no_stress_probability = probabilities[0][0]

prediction_label = (
    "Stress"
    if prediction[0] == 1
    else "No Stress"
)

print("\nPrediction :", prediction_label)

print(
    "Confidence :",
    f"{max(probabilities[0]) * 100:.2f}%"
)

print(
    "Stress Probability :",
    f"{stress_probability * 100:.2f}%"
)

print(
    "No Stress Probability :",
    f"{no_stress_probability * 100:.2f}%"
)