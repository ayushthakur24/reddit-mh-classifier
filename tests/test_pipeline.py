"""
Tests for the trained Scikit-Learn pipeline.
"""

import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ==========================================================
# Load Pipeline Once
# ==========================================================

pipeline = joblib.load(
    "model/stress_detection_pipeline.pkl"
)


# ==========================================================
# Pipeline Object
# ==========================================================

def test_pipeline_instance():

    assert isinstance(
        pipeline,
        Pipeline
    )


# ==========================================================
# TF-IDF Step Exists
# ==========================================================

def test_tfidf_exists():

    assert "tfidf" in pipeline.named_steps


# ==========================================================
# Classifier Exists
# ==========================================================

def test_classifier_exists():

    assert "classifier" in pipeline.named_steps


# ==========================================================
# TF-IDF Type
# ==========================================================

def test_tfidf_type():

    assert isinstance(
        pipeline.named_steps["tfidf"],
        TfidfVectorizer
    )


# ==========================================================
# Classifier Type
# ==========================================================

def test_classifier_type():

    assert isinstance(
        pipeline.named_steps["classifier"],
        LogisticRegression
    )