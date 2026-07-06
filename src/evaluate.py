"""
Evaluate the trained Reddit Mental Health Stress Detection model.

This script:
1. Loads the trained pipeline
2. Loads the test dataset
3. Generates predictions
4. Calculates evaluation metrics
5. Saves evaluation artifacts
"""

# ==========================================================
# Imports
# ==========================================================

import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.preprocess import clean_text

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# ==========================================================
# Project Constants
# ==========================================================

TEST_DATA_PATH = "data/raw/extracted/dreaddit-test.csv"

MODEL_PATH = "model/stress_detection_pipeline.pkl"

ASSETS_PATH = Path("assets")


# ==========================================================
# Load Test Dataset
# ==========================================================

test_df = pd.read_csv(TEST_DATA_PATH)

# Create a fresh DataFrame to avoid fragmentation warnings
test_df = test_df.copy()

test_df["clean_text"] = test_df["text"].apply(clean_text)

X_test = test_df["clean_text"]

y_test = test_df["label"]


# ==========================================================
# Load Trained Pipeline
# ==========================================================

pipeline = joblib.load(MODEL_PATH)


# ==========================================================
# Generate Predictions
# ==========================================================

predictions = pipeline.predict(X_test)

prediction_probabilities = pipeline.predict_proba(X_test)


# ==========================================================
# Calculate Metrics
# ==========================================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)

f1 = f1_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)

# ==========================================================
# Print Evaluation Metrics
# ==========================================================

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ==========================================================
# Save Metrics as JSON
# ==========================================================

metrics = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4)
}

with open(
    ASSETS_PATH / "metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ==========================================================
# Save Classification Report
# ==========================================================

report = classification_report(
    y_test,
    predictions
)

with open(
    ASSETS_PATH / "classification_report.txt",
    "w"
) as file:

    file.write(report)


# ==========================================================
# Confusion Matrix
# ==========================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Stress", "Stress"]
)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig(
    ASSETS_PATH / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nEvaluation artifacts saved successfully.")