import pandas as pd

#loading test dataset
test_df = pd.read_csv("data/raw/extracted/dreaddit-test.csv")

print("Test Dataset Shape:")
print(test_df.shape)

print("\nColumns:")
print(test_df.columns.to_list())

print("\nLabel Distribution:")
print(test_df["label"].value_counts())

import joblib

tfidf = joblib.load(
    "model/tkidf.pkl"
)
 
model = joblib.load(
    "model/logistic_regression.pkl"
)

print("Artifacts loaded successfully")

#Creating features from unseen test posts
X_test = test_df["text"]

#Actual labels
y_test = test_df["label"]

#Convert text using the same TF-IDF vocabulury learned during training
X_test_tfidf = tfidf.transform(
    X_test
)

#Predict labels
predictions = model.predict(
    X_test_tfidf
)

print("\nPredictions generated")
print("Number of predictions:", len(predictions))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

print("\nModel performance")
print("Accuracy :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall :", round(recall, 4))
print("F1 score :", round(f1, 4))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion matrix:\n")
print(cm)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Stress Detection Confusion Matrix")

plt.show()

