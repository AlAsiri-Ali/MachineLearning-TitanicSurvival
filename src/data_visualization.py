"""
Data visualization functions for the Titanic ML project.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_missing_values_heatmap(df: pd.DataFrame):
    """
    Creates a heatmap showing missing values in the dataset.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

def plot_survival_distribution(df: pd.DataFrame):
    """
    Plots the distribution of survival outcomes.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Survived', data=df)
    plt.title('Distribution of Survival Outcome (0 = No, 1 = Yes)')
    plt.xlabel('Survival Status')
    plt.ylabel('Number of Passengers')
    plt.xticks(ticks=[0, 1], labels=['Did Not Survive', 'Survived'])
    plt.show()

def plot_survival_by_sex(df: pd.DataFrame):
    """
    Plots survival count by sex.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Sex', hue='Survived')
    plt.title('Survival Count by Sex')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.show()

def plot_survival_by_pclass(df: pd.DataFrame):
    """
    Plots survival count by passenger class.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='magma')
    plt.title('Survival Count by Passenger Class')
    plt.xlabel('Passenger Class (Pclass)')
    plt.ylabel('Number of Passengers')
    plt.legend(title='Survived', labels=['Did Not Survive', 'Survived'])
    plt.show()

def plot_age_distribution_by_survival(df: pd.DataFrame):
    """
    Plots age distribution by survival status.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='Age', hue='Survived', kde=True)
    plt.title('Age Distribution by Survival Status')
    plt.xlabel('Age (Capped)')
    plt.ylabel('Frequency')
    plt.legend(title='Survived', labels=['Did Not Survive', 'Survived'])
    plt.show()

def plot_fare_boxplot_by_survival(df: pd.DataFrame):
    """
    Plots fare distribution by survival status using boxplot.
    """
    plt.figure(figsize=(8, 6))
    fare_col = 'Fare_log' if 'Fare_log' in df.columns else 'Fare'
    fare_lbl = 'Fare (Log Transformed)' if fare_col == 'Fare_log' else 'Fare (Original)'
    
    sns.boxplot(data=df, x='Survived', y=fare_col, hue='Survived', legend=False)
    plt.title(f'{fare_lbl} Box Plot by Survival')
    plt.ylabel(fare_lbl)
    plt.xlabel('Survival Status')
    plt.xticks(ticks=[0, 1], labels=['Did Not Survive', 'Survived'])
    plt.show()

def plot_survival_by_embarked(df: pd.DataFrame):
    """
    Plots survival count by port of embarkation.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Embarked', hue='Survived', data=df, palette='rocket')
    plt.title('Survival Count by Port of Embarkation')
    plt.xlabel('Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)')
    plt.ylabel('Number of Passengers')
    plt.legend(title='Survived', labels=['Did Not Survive', 'Survived'])
    plt.show()

def plot_numerical_features_boxplots(df: pd.DataFrame):
    """
    Creates box plots for numerical features to identify outliers.
    """
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
    
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(numerical_features):
        if feature in df.columns:
            plt.subplot(2, 2, i + 1)
            sns.boxplot(x=df[feature])
            plt.title(f'Box Plot of {feature}')
    
    plt.tight_layout()
    plt.show()

def plot_numerical_features_histograms(df: pd.DataFrame):
    """
    Creates histograms for numerical features.
    """
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
    
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(numerical_features):
        if feature in df.columns:
            plt.subplot(2, 2, i + 1)
            plt.hist(df[feature].dropna(), bins=20, edgecolor='black')
            plt.title(f'Histogram of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def create_comprehensive_eda_report(df: pd.DataFrame):
    """
    Creates a comprehensive EDA report with all visualizations.
    """
    print("=== COMPREHENSIVE EDA REPORT ===\n")
    
    print("1. Missing Values Analysis")
    plot_missing_values_heatmap(df)
    
    print("\n2. Survival Distribution")
    plot_survival_distribution(df)
    
    print("\n3. Survival by Gender")
    plot_survival_by_sex(df)
    
    print("\n4. Survival by Passenger Class")
    plot_survival_by_pclass(df)
    
    print("\n5. Age Distribution by Survival")
    plot_age_distribution_by_survival(df)
    
    print("\n6. Fare Distribution by Survival")
    plot_fare_boxplot_by_survival(df)
    
    print("\n7. Survival by Embarkation Port")
    plot_survival_by_embarked(df)
    
    print("\n8. Outlier Analysis - Box Plots")
    plot_numerical_features_boxplots(df)
    
    print("\n9. Distribution Analysis - Histograms")
    plot_numerical_features_histograms(df)
    
    print("\n=== EDA REPORT COMPLETE ===")
