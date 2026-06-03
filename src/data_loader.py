import pandas as pd

def load_train_data():
    return pd.read_csv("../data/raw/extracted/dreaddit-train.csv")


def load_test_data():
    return pd.read_csv("../data/raw/extracted/dreaddit-test.csv")