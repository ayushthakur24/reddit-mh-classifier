# Engineering Challenges Faced

This document records the engineering problems encountered while developing the Reddit Mental Health Stress Detection project and how each issue was resolved.

---

# 1. Dataset Management

Problem

Large datasets should not be committed to GitHub.

Solution

Configured GitHub Actions to download the Kaggle dataset during every CI run using Kaggle API credentials stored as GitHub Secrets.

---

# 2. Reproducible Pipeline

Problem

Training, prediction and evaluation used different preprocessing logic.

Solution

Centralized preprocessing into preprocess.py so every stage uses identical cleaning logic.

---

# 3. Missing Model Artifacts

Problem

GitHub Actions initially failed because model files did not exist.

Solution

Modified the CI pipeline to train the model before executing tests.

---

# 4. GitHub Actions Failures

Problems solved included:

- Missing datasets
- Missing processed CSV files
- Missing model artifacts
- Incorrect file paths
- Linux path compatibility
- Missing Kaggle authentication

---

# 5. Prediction Consistency

Ensured preprocessing is identical for:

- Training
- Evaluation
- Inference
- Streamlit application

---

# 6. Evaluation Artifacts

Implemented automatic generation of:

- Metrics JSON
- Classification Report
- Confusion Matrix
- ROC Curve
- Precision Recall Curve

---

# 7. Unit Testing

Created 24 automated unit tests covering:

- Text preprocessing
- Pipeline integrity
- Prediction correctness
- Generated artifacts

---

# 8. CI/CD Automation

Every push now automatically:

- Downloads dataset
- Preprocesses text
- Trains model
- Evaluates model
- Generates visualizations
- Runs unit tests

No manual intervention is required.

---

# Key Takeaway

The biggest engineering lesson was learning how to make an ML project reproducible from a completely clean environment.