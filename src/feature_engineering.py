"""
Feature engineering functions for the Titanic ML project.
"""
import pandas as pd
import numpy as np


def create_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds FamilySize and IsAlone features to the DataFrame.
    FamilySize = SibSp + Parch + 1
    IsAlone = 1 if FamilySize == 1 else 0
    """
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df


def bin_age(df: pd.DataFrame, age_col: str = 'Age') -> pd.DataFrame:
    """
    Bins the Age column into categorical bins and adds an AgeBin column.
    Bins: Child, Teenager, Young Adult, Adult, Senior
    """
    df = df.copy()
    bins = [0, 12, 18, 30, 50, np.inf]
    labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
    df['AgeBin'] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
    return df


def bin_fare(df: pd.DataFrame, fare_col: str = 'Fare_log') -> pd.DataFrame:
    """
    Bins the Fare_log column into four quantiles and adds a FareBin column.
    Bins: Low, Medium, High, Very High
    """
    df = df.copy()
    labels = ['Low', 'Medium', 'High', 'Very High']
    df['FareBin'] = pd.qcut(df[fare_col], q=4, labels=labels)
    return df


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps in sequence.
    """
    df = create_family_features(df)
    df = bin_age(df)
    df = bin_fare(df)
    return df