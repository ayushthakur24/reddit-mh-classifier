# ==========================================================
# Imports
# ==========================================================

import joblib
import matplotlib.pyplot as plt

# ==========================================================
# Load Pipeline
# ==========================================================

pipeline = joblib.load(
    "model/stress_detection_pipeline.pkl"
)

tfidf = pipeline.named_steps["tfidf"]

classifier = pipeline.named_steps["classifier"]

# ==========================================================
# Extract Feature Importance
# ==========================================================

feature_names = tfidf.get_feature_names_out()

coefficients = classifier.coef_[0]

# ==========================================================
# Top Stress Words
# ==========================================================

top_positive_indices = coefficients.argsort()[-20:]

positive_words = [
    feature_names[index]
    for index in top_positive_indices
]

positive_scores = [
    coefficients[index]
    for index in top_positive_indices
]

plt.figure(figsize=(10,7))

plt.barh(
    positive_words,
    positive_scores
)

plt.title("Top 20 Stress Indicators")

plt.xlabel("Coefficient")

plt.tight_layout()

plt.savefig(
    "assets/top_stress_words.png",
    dpi=300
)

plt.close()

# ==========================================================
# Top No Stress Words
# ==========================================================

top_negative_indices = coefficients.argsort()[:20]

negative_words = [
    feature_names[index]
    for index in top_negative_indices
]

negative_scores = [
    abs(coefficients[index])
    for index in top_negative_indices
]

plt.figure(figsize=(10,7))

plt.barh(
    negative_words,
    negative_scores
)

plt.title("Top 20 No Stress Indicators")

plt.xlabel("Coefficient")

plt.tight_layout()

plt.savefig(
    "assets/top_no_stress_words.png",
    dpi=300
)

plt.close()

print("\nVisualization artifacts generated successfully.")