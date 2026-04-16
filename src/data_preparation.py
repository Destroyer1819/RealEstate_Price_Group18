"""
data_preparation.py
-------------------
Loads, cleans, encodes, and splits the Melbourne Housing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'melb_data.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

TARGET = 'Price'
FEATURES = ['Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'YearBuilt', 'Regionname']


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df[FEATURES + [TARGET]].copy()
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Removed {before - after} rows with missing values. Rows remaining: {after}")
    df = df.drop_duplicates()
    print(f"Rows after deduplication: {len(df)}")
    return df


def prepare_features(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_prepared_data():
    df = load_data()
    df = clean_data(df)
    X_encoded, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X_encoded, y)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(list(X_train.columns), os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    get_prepared_data()
    print("Data preparation complete.")
