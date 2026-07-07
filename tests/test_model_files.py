"""
Tests to verify required project artifacts exist.
"""

from pathlib import Path


# ==========================================================
# Project Paths
# ==========================================================

MODEL_DIR = Path("model")

ASSETS_DIR = Path("assets")


# ==========================================================
# Pipeline Exists
# ==========================================================

def test_pipeline_exists():

    assert (
        MODEL_DIR /
        "stress_detection_pipeline.pkl"
    ).exists()


# ==========================================================
# Metrics JSON Exists
# ==========================================================

def test_metrics_exists():

    assert (
        ASSETS_DIR /
        "metrics.json"
    ).exists()


# ==========================================================
# Classification Report Exists
# ==========================================================

def test_classification_report_exists():

    assert (
        ASSETS_DIR /
        "classification_report.txt"
    ).exists()


# ==========================================================
# Confusion Matrix Exists
# ==========================================================

def test_confusion_matrix_exists():

    assert (
        ASSETS_DIR /
        "confusion_matrix.png"
    ).exists()


# ==========================================================
# ROC Curve Exists
# ==========================================================

def test_roc_curve_exists():

    assert (
        ASSETS_DIR /
        "roc_curve.png"
    ).exists()


# ==========================================================
# Precision Recall Curve Exists
# ==========================================================

def test_precision_recall_curve_exists():

    assert (
        ASSETS_DIR /
        "precision_recall_curve.png"
    ).exists()


# ==========================================================
# Top Stress Words Visualization Exists
# ==========================================================

def test_top_stress_words_exists():

    assert (
        ASSETS_DIR /
        "top_stress_words.png"
    ).exists()


# ==========================================================
# Top No Stress Words Visualization Exists
# ==========================================================

def test_top_no_stress_words_exists():

    assert (
        ASSETS_DIR /
        "top_no_stress_words.png"
    ).exists()