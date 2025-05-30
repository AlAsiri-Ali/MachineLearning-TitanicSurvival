# 🚢 Titanic Survival Prediction - Master's Project

![University](https://img.shields.io/badge/University-King%20Abdulaziz%20University-blue)
![College](https://img.shields.io/badge/College-FCIT-green)
![Masters Project](https://img.shields.io/badge/Level-Masters%20Degree-blue)
![Subject](https://img.shields.io/badge/Subject-Machine%20Learning-orange)
![Language](https://img.shields.io/badge/Language-Python-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 🎓 Academic Information

**University:** King Abdulaziz University<br>
**College:** Faculty of Computing and Information Technology (FCIT)<br>
**Department:** Computer Science<br>
**Academic Level:** Masters Degree<br>
**Course:** Machine Learning<br>

## 🎯 Project Overview

This project analyzes the famous Titanic dataset to predict passenger survival using machine learning techniques. The analysis employs **supervised learning** for **binary classification** to determine factors that influenced survival during the disaster.

### 🔍 Problem Framing
- **Problem Domain:** Survival prediction analysis
- **Machine Learning Style:** Supervised Learning
- **Model Type:** Binary Classification
- **Target Variable:** Survived (0 = Did not survive, 1 = Survived)
- **Dataset:** Titanic - Machine Learning from Disaster (Kaggle)

## 📋 Quick Links

- **[📊 Jupyter Notebook](notebooks/Machen_learning_project.ipynb)** - Complete analysis and model development
- **[🐍 Python Script](notebooks/Machen_learning_project.py)** - Converted notebook to Python script
- **[🚀 Main Pipeline](main.py)** - Automated ML pipeline execution

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (for interactive analysis)

### 2. Installation

1. **Clone or download the project:**
   ```bash
   git clone https://github.com/AlAsiri-Ali/MachineLearning-TitanicSurvival.git

   cd MachineLearning-TitanicSurvival
   ```

2. **Install dependencies:**
   ```bash
    pip install -r requirements.txt
   ```

3. **Data is included:**
   - `data/train.csv` - Training dataset (891 passengers)
   - `data/test.csv` - Test dataset for predictions

### 3. Running the Analysis

**Option 1: Run the complete ML pipeline**
```bash
python main.py
```

**Option 2: Use individual modules**
```python
from src.data_preprocessing import load_data, preprocessing_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.model_training import train_logistic_regression, train_svc
from src.model_evaluation import evaluate_model
from src.data_visualization import create_comprehensive_eda_report

# Load and process data
df = load_data("data/train.csv")

# Optional: Run comprehensive EDA
create_comprehensive_eda_report(df)

# Process data
df = preprocessing_pipeline(df)
df = feature_engineering_pipeline(df)
```

**Option 3: Explore the Jupyter Notebook**
```bash
jupyter notebook "notebooks/Machen_learning_project.ipynb"
```

## 📊 Analysis Pipeline

The project follows a comprehensive machine learning pipeline:

### 1. 🔍 Data Loading and Exploration
- Load training dataset (891 passengers, 12 features)
- Examine data structure, types, and basic statistics
- Identify numerical and categorical features
- Generate comprehensive data summary

### 2. 🔍 Missing Values Analysis
- **Age:** 177 missing values (~19.9%)
- **Cabin:** 687 missing values (~77.1%)
- **Embarked:** 2 missing values (~0.2%)
- Create visualizations (heatmaps) to understand patterns

### 3. 🧹 Data Preprocessing
- **Age:** Group-based median imputation (by Pclass and Title)
- **Embarked:** Mode imputation (filled with 'S')
- **Cabin:** Extract deck information, create 'Deck' feature
- **Title:** Extract from Name column (Mr, Mrs, Miss, Master, Rare)

### 4. 🔧 Outlier Detection and Handling
- **Age:** Capped at 99th percentile (65.0 years)
- **Fare, SibSp, Parch:** Outliers retained as genuine data
- Box plots and histograms for visualization

### 5. 📈 Exploratory Data Analysis (EDA)
- Survival distribution analysis
- Survival patterns by Sex, Pclass, Age, Embarked
- Feature correlation analysis
- Comprehensive visualizations (pair plots, count plots, histograms)

### 6. ⚙️ Feature Engineering
- **FamilySize:** SibSp + Parch + 1
- **IsAlone:** Binary feature (1 if FamilySize == 1)
- **Fare_log:** Log transformation of Fare
- **AgeBin:** Age categorization (Child, Teenager, Young Adult, Adult, Senior)
- **FareBin:** Fare quantile binning (Low, Medium, High, Very High)

### 7. 🎯 Encoding and Scaling
- **Sex:** Label encoding (Female: 0, Male: 1)
- **Embarked, Title, Deck:** One-hot encoding
- **AgeBin, FareBin:** Ordinal encoding
- **Numerical features:** StandardScaler normalization

### 8. 🤖 Model Training and Evaluation
- **Models:** Logistic Regression, SVC, Random Forest
- **Hyperparameter tuning:** GridSearchCV with 5-fold cross-validation
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score
- **Visualization:** Confusion matrix, ROC curves

## 📊 Key Features Analyzed

### Original Features
- **PassengerId:** Unique identifier
- **Survived:** Target variable (0/1)
- **Pclass:** Passenger class (1st, 2nd, 3rd)
- **Name:** Passenger name (used for title extraction)
- **Sex:** Gender (male, female)
- **Age:** Passenger age in years
- **SibSp:** Number of siblings/spouses aboard
- **Parch:** Number of parents/children aboard
- **Ticket:** Ticket number
- **Fare:** Ticket fare
- **Cabin:** Cabin number (high missing rate)
- **Embarked:** Port of embarkation (S, C, Q)

### Engineered Features
- **Title:** Extracted from Name (Mr, Mrs, Miss, Master, Rare)
- **Deck:** Extracted from Cabin (A-G, M for missing)
- **FamilySize:** Total family members aboard
- **IsAlone:** Binary indicator for solo travelers
- **Fare_log:** Log-transformed fare
- **AgeBin:** Age categories
- **FareBin:** Fare quartiles

## 🔧 Project Structure

```
├── data/
│   ├── train.csv .............................. # Training dataset
│   └── test.csv  .............................. # Test dataset
├── notebooks/
│   ├── Machen_learning_project.ipynb .......... # Main Jupyter notebook
│   ├── Machen_learning_project.py ............. # Converted Python script
│   └── Machen_learning_project_Report.md ...... # Project report
├── src/
│   ├── data_preprocessing.py .................. # Data loading and preprocessing
│   ├── feature_engineering.py ................. # Feature creation and transformation
│   ├── model_training.py ...................... # ML model training functions
│   ├── model_evaluation.py .................... # Model evaluation and visualization
│   ├── data_visualization.py .................. # EDA and data visualization functions
│   └── utils.py ............................... # Utility functions
├── main.py .................................... # Main pipeline execution
├── model.jobli ................................ # Saved best model
└── README.md .................................. # Project documentation
```

## 📈 Key Insights

Based on the comprehensive analysis, key factors affecting survival include:

1. **Gender:** Females had significantly higher survival rates (~74% vs ~19% for males)
2. **Passenger Class:** Clear hierarchy - 1st class (highest survival), 2nd class (moderate), 3rd class (lowest)
3. **Age:** Children (under 10) had higher survival rates; young adults (20-35) had lower rates
4. **Family Size:** Passengers with small families (1-3 members) had better survival chances than solo travelers or large families
5. **Embarkation Port:** Cherbourg (C) passengers had highest survival rate, followed by Queenstown (Q) and Southampton (S)
6. **Fare:** Higher fare passengers (correlated with class) had better survival chances

## 🏆 Model Performance

### Final Results (Test Set)
- **Optimized Logistic Regression:** 83.8% accuracy, F1-score: 0.79
- **Optimized SVC:** 83.8% accuracy, F1-score: 0.78
- **Random Forest:** Competitive performance with feature importance insights

### Best Parameters
- **Logistic Regression:** C=10, penalty='l1', solver='liblinear'
- **SVC:** C=1, gamma='scale', kernel='rbf'

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas:** Data manipulation and analysis
- **numpy:** Numerical computing
- **matplotlib/seaborn:** Data visualization
- **scikit-learn:** Machine learning algorithms and utilities
- **imbalanced-learn:** SMOTE for handling class imbalance
- **joblib:** Model serialization
- **jupyter:** Interactive development and analysis

## 📝 Usage Examples

### Loading and Preprocessing Data
```python
from src.data_preprocessing import load_data, preprocessing_pipeline

# Load raw data
df = load_data("data/train.csv")

# Apply preprocessing pipeline
df_processed = preprocessing_pipeline(df)
print(f"Dataset shape after preprocessing: {df_processed.shape}")
```

### Feature Engineering
```python
from src.feature_engineering import feature_engineering_pipeline

# Apply feature engineering
df_engineered = feature_engineering_pipeline(df_processed)

# Check new features
print("New features created:")
print(df_engineered.columns.tolist())
```

### Model Training
```python
from src.model_training import train_logistic_regression, train_svc
from src.utils import split_data

# Prepare data
X = df_engineered.drop(['Survived'], axis=1)
y = df_engineered['Survived']
X_train, X_test, y_train, y_test = split_data(X, y)

# Train models
lr_model = train_logistic_regression(X_train, y_train, grid_search=True)
svc_model = train_svc(X_train, y_train, grid_search=True)
```

### Model Evaluation
```python
from src.model_evaluation import evaluate_model, plot_confusion_matrix

# Evaluate model
accuracy, report = evaluate_model(lr_model, X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Plot confusion matrix
plot_confusion_matrix(lr_model, X_test, y_test)
```

### Data Visualization and EDA
```python
from src.data_visualization import create_comprehensive_eda_report

# Run comprehensive exploratory data analysis
create_comprehensive_eda_report(df)

# Or use individual visualization functions
from src.data_visualization import plot_survival_by_sex, plot_survival_by_pclass
plot_survival_by_sex(df)
plot_survival_by_pclass(df)
```

## 📚 Learning Outcomes

This project demonstrates mastery of:

- **Data Science Workflow:** Complete pipeline from raw data to model deployment
- **Data Preprocessing:** Handling missing values, outliers, and data quality issues
- **Feature Engineering:** Creating meaningful features from raw data
- **Exploratory Data Analysis:** Comprehensive data visualization and pattern discovery
- **Machine Learning:** Supervised learning, classification algorithms, hyperparameter tuning
- **Model Evaluation:** Performance metrics, cross-validation, model comparison
- **Code Organization:** Modular programming, reproducible research practices

---

**Note:** This project demonstrates the complete machine learning workflow from data exploration to model evaluation, following academic best practices for reproducible research and code organization. The analysis provides valuable insights into the factors that influenced survival during the Titanic disaster.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Titanic dataset provided by [Kaggle](https://www.kaggle.com/c/titanic).
- Developed as part of the Machine Learning course at King Abdulaziz University (EMAI640).
- Thanks to instructors, colleagues, and open-source contributors whose work inspired this project.
