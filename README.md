# reddit-mh-classifier
A machine learning classifier that detects mental health distress signals in Reddit posts using NLP. Built with Scikit-learn, deployed on Streamlit Community Cloud.


Problem Statement
Reddit communities like r/depression and r/anxiety contain posts where users express distress — but these signals are hard to surface at scale. This project trains a binary text classifier to flag posts that show signs of mental health distress, using only the post text as input.


Dataset

Source: Stress Analysis in Social Media — Kaggle
Size: ~50,000 Reddit posts
Labels: 1 = distress signal, 0 = no distress signal
Subreddits covered: r/depression, r/anxiety, r/SuicideWatch (positive class) vs neutral subreddits (negative class)

Tech Stack
Layer                Tool
Language             Python 3.11
Data processing      Pandas,NLTK
Feature engineering  TF-IDF(scikit-learn)
Model                Logistic Regression (scikit-learn Pipeline)
Evaluation           F1 score, confusion matrix, cross-validation
App                  Streamlit
Deployment           Streamlit Community Cloud
Version control      Git + GitHub


How It Works
Raw post text
     │
     ▼
Text cleaning        lowercase, strip URLs, remove punctuation
     │
     ▼
TF-IDF vectorizer    converts text → sparse numerical matrix (bigrams, top 10k features)
     │
     ▼
Logistic Regression  binary classification with class_weight='balanced'
     │
     ▼
Prediction + confidence score


Results
Metric                  Score
CV F1 (weighted)        ~0.87
Test Accuracy           ~88%
Test F1 (weighted)      ~0.87

Scores are approximate — retrain to reproduce exact numbers.


Disclaimer
This project is for educational purposes only. If you or someone you know is experiencing a mental health crisis, please contact a qualified professional or a crisis helpline.
