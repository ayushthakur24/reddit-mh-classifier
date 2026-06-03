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

Day 1 Summary

On Day 1, I set up the project structure, configured the Python virtual environment, connected the Kaggle API, and downloaded the Dreaddit Stress Detection dataset. The dataset contained separate training and testing files (`dreaddit-train.csv` and `dreaddit-test.csv`), which follows a research-grade train/test split approach and eliminates the need for `train_test_split()` during initial model development.

I performed Exploratory Data Analysis (EDA) using Pandas, Matplotlib, and Seaborn. The training dataset contains 2,838 Reddit posts with a nearly balanced label distribution (1,488 stress posts and 1,350 non-stress posts), making it suitable for classification without additional balancing techniques. I analyzed the dataset schema, inspected sample records, checked for missing values, measured post lengths, and generated visualizations to understand data quality and distribution.

Further analysis revealed that URLs appeared in only one post across the entire dataset, indicating that aggressive text cleaning was unnecessary. Based on this observation, I implemented a lightweight preprocessing strategy that preserves punctuation, sentence structure, contractions, and emotional signals while converting text to lowercase and normalizing whitespace. This approach is important because linguistic cues often carry predictive information in mental health and stress detection tasks.

The processed dataset was saved as `train_clean.csv` and will be used in subsequent stages for TF-IDF feature extraction, model training, evaluation, and deployment.
