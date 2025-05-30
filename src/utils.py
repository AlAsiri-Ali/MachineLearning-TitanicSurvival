"""
Utility functions for the Titanic ML project.
"""
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
import joblib

def set_seed(seed: int = 42):
    """
    Sets random seed for reproducibility across numpy, random, and os environments.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Splits features and target into train and test sets.
    Returns X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def save_model(model, path: str):
    """
    Saves a trained model to disk using joblib.
    """
    joblib.dump(model, path)

def load_model(path: str):
    """
    Loads a model from disk using joblib.
    """
    return joblib.load(path)