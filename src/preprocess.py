"""
Preprocess the Reddit Mental Health dataset.

This module:
1. Cleans Reddit posts
2. Generates processed training dataset
"""

# ==========================================================
# Imports
# ==========================================================

import re
from pathlib import Path

import pandas as pd


# ==========================================================
# Project Paths
# ==========================================================

RAW_DATA_PATH = Path("data/raw/extracted/dreaddit-train.csv")

PROCESSED_DIR = Path("data/processed")

PROCESSED_FILE = PROCESSED_DIR / "train_clean.csv"


# ==========================================================
# Text Cleaning
# ==========================================================

def clean_text(text):
    """
    Cleans Reddit posts before passing them to the ML model.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove punctuation
    4. Remove extra spaces
    """

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# Dataset Preprocessing
# ==========================================================

def preprocess_dataset():
    """
    Loads the raw dataset, cleans the text,
    and saves the processed dataset.
    """

    print("Loading training dataset...")

    train_df = pd.read_csv(RAW_DATA_PATH)

    print(f"Loaded {len(train_df)} records.")

    train_df = train_df.copy()

    train_df["clean_text"] = train_df["text"].apply(clean_text)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print(f"\nProcessed dataset saved to:\n{PROCESSED_FILE}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    preprocess_dataset()