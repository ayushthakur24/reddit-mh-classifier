# ==========================================================
# Imports
# ==========================================================
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================================
# Load Training Dataset
# ==========================================================
train_df = pd.read_csv("data/processed/train_clean.csv")


# ==========================================================
# Prepare Features and Target Variable
# ==========================================================
X = train_df["clean_text"]
y = train_df["label"]


# ==========================================================
# Build Machine Learning Pipeline
# ==========================================================
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# Train the complete pipeline.
# This internally performs:
# 1. TF-IDF vocabulary learning
# 2. Text vectorization
# 3. Logistic Regression training
pipeline.fit(
    X,
    y
)


# ==========================================================
# Retrieve Trained Pipeline Components
# ==========================================================

tfidf = pipeline.named_steps["tfidf"]

classifier = pipeline.named_steps["classifier"]

X_tfidf = tfidf.transform(X)

print("TF-IDF Matrix Shape:")
print(X_tfidf.shape)

print("\nVocabulary Size:")
print(len(tfidf.vocabulary_))

feature_names = tfidf.get_feature_names_out()

print("\nFeatures 1000 to 1050:\n")

for word in feature_names[1000:1050]: 
    print(word)


print("\nModel training completed")

# ==========================================================
# Save Trained Pipeline
# ==========================================================

joblib.dump(
    pipeline,
    "model/stress_detection_pipeline.pkl"
)

print("\nPipeline saved successfully.")

# ==========================================================
# Analyze Feature Importance
# ==========================================================

coefficients = classifier.coef_[0]

top_positive = coefficients.argsort()[-20:]

print("\nTop Stress Indicators:\n")

for index in reversed(top_positive):
    print(
        feature_names[index],
        round(coefficients[index],4)
    )

