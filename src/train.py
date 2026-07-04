import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

#load processed dataset
train_df = pd.read_csv("data/processed/train_clean.csv")

X = train_df["clean_text"]
y = train_df["label"]

#Creating TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)

#Converting text into Vectors
X_tfidf = tfidf.fit_transform(X)

print("TF-IDF Matrix Shape:")
print(X_tfidf.shape)

print("\nVocabulary Size:")
print(len(tfidf.vocabulary_))

feature_names = tfidf.get_feature_names_out()

print("\nFeatures 1000 to 1050:\n")

for word in feature_names[1000:1050] : 
    print(word)


from sklearn.linear_model import LogisticRegression

#Train classifier
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_tfidf,
    y
)

print("\nModel training completed")

import joblib

joblib.dump(
    tfidf,
    "model/tkidf.pkl"
)

joblib.dump(
    model,
    "model/logistic_regression.pkl"
)

print("Model and vectorizer saved.")

#Trying to understand words making model predict stree
#Using Model Interpretability

feature_names = tfidf.get_feature_names_out()

coefficients = model.coef_[0]

top_positive = coefficients.argsort()[-20:]

print("\nTop Stress Indicators:\n")

for index in reversed(top_positive):
    print(
        feature_names[index],
        round(coefficients[index],4)
    )

