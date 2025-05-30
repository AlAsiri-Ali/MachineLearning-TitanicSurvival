"""
Data preprocessing functions for the Titanic ML project.
"""
import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    """
    Loads Titanic data from a CSV file.
    """
    return pd.read_csv(path)

def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the title from the Name column and adds it as a new column 'Title'.
    """
    df = df.copy()
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    return df

def simplify_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplifies rare titles into a single 'Rare' category and standardizes common ones.
    """
    df = df.copy()
    title_map = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
    }
    df['Title'] = df['Title'].replace(title_map)
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    return df

def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing Age values using the median age for each Pclass and Title group.
    """
    df = df.copy()
    # Calculate median ages per group
    median_ages = df.groupby(['Pclass', 'Title'])['Age'].median()

    # Function to impute age based on Pclass and Title
    def impute_age_row(row):
        if pd.isnull(row['Age']):
            try:
                return median_ages.loc[row['Pclass'], row['Title']]
            except KeyError:
                # If combination doesn't exist, use overall median
                return df['Age'].median()
        return row['Age']

    # Apply the function to each row
    df['Age'] = df.apply(impute_age_row, axis=1)
    return df

def extract_deck(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the deck from the Cabin column and adds it as a new column 'Deck'.
    Missing values are filled with 'M' (for 'Missing').
    """
    df = df.copy()
    df['Deck'] = df['Cabin'].str[0].fillna('M')
    return df

def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in the Embarked column with the mode.
    """
    df = df.copy()
    mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(mode)
    return df

def cap_age(df: pd.DataFrame, cap: float = 65.0) -> pd.DataFrame:
    """
    Caps the Age column at a specified value (default 65.0).
    """
    df = df.copy()
    df['Age'] = np.where(df['Age'] > cap, cap, df['Age'])
    return df

def log_transform_fare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies log1p transformation to the Fare column and adds a new column 'Fare_log'.
    """
    df = df.copy()
    df['Fare_log'] = np.log1p(df['Fare'])
    return df

def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all preprocessing steps in sequence.
    """
    df = extract_title(df)
    df = simplify_title(df)
    df = impute_age(df)
    df = extract_deck(df)
    df = impute_embarked(df)
    df = cap_age(df)
    df = log_transform_fare(df)
    return df