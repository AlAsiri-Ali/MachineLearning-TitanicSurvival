from src.data_preprocessing import load_data, preprocessing_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.utils import set_seed, split_data, save_model
from src.model_training import train_logistic_regression, train_svc, train_random_forest
from src.model_evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve
from src.data_visualization import create_comprehensive_eda_report
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -- Set random seed for reproducibility
set_seed(42)

# -- Load data
train_path = "data/train.csv"
df = load_data(train_path)

# -- Optional: Run comprehensive EDA (uncomment to enable)
# print("[INFO] Running Exploratory Data Analysis...")
# create_comprehensive_eda_report(df)

# --Data preprocessing
df = preprocessing_pipeline(df)

# -- Feature engineering
df = feature_engineering_pipeline(df)

# -- Encoding categorical variables

# Label Encoding for 'Sex'
sex_mapping = {'female': 0, 'male': 1}
df['Sex'] = df['Sex'].map(sex_mapping)

# One-Hot Encoding for 'Embarked', 'Title', 'Deck'
df = pd.get_dummies(df, columns=['Embarked', 'Title', 'Deck'], drop_first=True)

# Ordinal Encoding for 'AgeBin', 'FareBin'
age_order = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
fare_order = ['Low', 'Medium', 'High', 'Very High']
age_mapping_ordinal = {category: i for i, category in enumerate(age_order)}
fare_mapping_ordinal = {category: i for i, category in enumerate(fare_order)}
df['AgeBin'] = df['AgeBin'].map(age_mapping_ordinal)
df['FareBin'] = df['FareBin'].map(fare_mapping_ordinal)

# -- Prepare features and target
drop_cols = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']
X = df.drop(drop_cols, axis=1, errors='ignore')
y = df['Survived']

# -- Scaling numerical features
numerical_cols = ['Age', 'Fare_log', 'FamilySize', 'SibSp', 'Parch']
scaler = StandardScaler()
for col in numerical_cols:
    if col in X.columns:
        X[col] = scaler.fit_transform(X[[col]])

# -- Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# -- Train and compare multiple models
models = {
    'Logistic Regression': train_logistic_regression(X_train, y_train, grid_search=True),
    'SVC': train_svc(X_train, y_train, grid_search=True),
    'Random Forest': train_random_forest(X_train, y_train, grid_search=True)
}

results = {}
for name, model in models.items():
    print(f"\n[INFO] Model used: {name} ({type(model).__name__})")
    if hasattr(model, 'best_params_'):
        print(f"[INFO] Best parameters from GridSearchCV: {model.get_params()}")
    acc, _ = evaluate_model(model, X_test, y_test)
    results[name] = acc

# Identify and show the best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n[RESULT] Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# -- Evaluate and plot for the best model
plot_confusion_matrix(best_model, X_test, y_test)
plot_roc_curve(best_model, X_test, y_test)

# -- Save best model
save_model(best_model, "model.joblib")
