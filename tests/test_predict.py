"""
Tests for model prediction.
"""

import joblib
import numpy as np


# ==========================================================
# Load Pipeline
# ==========================================================

pipeline = joblib.load(
    "model/stress_detection_pipeline.pkl"
)


# ==========================================================
# Sample Inputs
# ==========================================================

stress_text = (
    "I feel hopeless and depressed."
)

non_stress_text = (
    "I had a wonderful vacation with my family."
)


# ==========================================================
# Prediction Returns Single Value
# ==========================================================

def test_single_prediction():

    prediction = pipeline.predict(
        [stress_text]
    )

    assert len(prediction) == 1


# ==========================================================
# Prediction Is Binary
# ==========================================================

def test_prediction_binary():

    prediction = pipeline.predict(
        [stress_text]
    )[0]

    assert prediction in [0, 1]


# ==========================================================
# Probability Shape
# ==========================================================

def test_probability_shape():

    probabilities = pipeline.predict_proba(
        [stress_text]
    )

    assert probabilities.shape == (1, 2)


# ==========================================================
# Probabilities Sum To One
# ==========================================================

def test_probability_sum():

    probabilities = pipeline.predict_proba(
        [stress_text]
    )[0]

    assert np.isclose(
        probabilities.sum(),
        1.0
    )


# ==========================================================
# Prediction Works On Non Stress Text
# ==========================================================

def test_non_stress_prediction():

    prediction = pipeline.predict(
        [non_stress_text]
    )

    assert len(prediction) == 1