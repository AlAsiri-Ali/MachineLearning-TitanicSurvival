"""
Model training functions for the Titanic ML project.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Any

def train_logistic_regression(X_train, y_train, grid_search: bool = False) -> Any:
    """
    Trains a Logistic Regression model. If grid_search=True, performs hyperparameter tuning.
    Returns the trained model (or best estimator).
    """
    if grid_search:
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        return grid.best_estimator_
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        return model

def train_svc(X_train, y_train, grid_search: bool = False) -> Any:
    """
    Trains a Support Vector Classifier. If grid_search=True, performs hyperparameter tuning.
    Returns the trained model (or best estimator).
    """
    if grid_search:
        param_grid = [
            {'C': [0.1, 1, 10, 50], 'kernel': ['linear']},
            {'C': [0.1, 1, 10, 50], 'kernel': ['rbf'], 'gamma': [0.1, 0.01, 0.001, 'scale']}
        ]
        grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        return grid.best_estimator_
    else:
        model = SVC(probability=True, random_state=42)
        model.fit(X_train, y_train)
        return model

def train_random_forest(X_train, y_train, grid_search: bool = False) -> Any:
    """
    Trains a Random Forest Classifier. If grid_search=True, performs hyperparameter tuning.
    Returns the trained model (or best estimator).
    """
    if grid_search:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        return grid.best_estimator_
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model