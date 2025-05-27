# 🚢 Titanic Survival Analysis Project

![University](https://img.shields.io/badge/University-King%20Abdulaziz%20University-blue)
![College](https://img.shields.io/badge/College-FCIT-green)
![Masters Project](https://img.shields.io/badge/Level-Masters%20Degree-blue)
![Course](https://img.shields.io/badge/Course-EMAI%20640-green)
![Subject](https://img.shields.io/badge/Subject-Machine%20Learning-orange)
![Language](https://img.shields.io/badge/Language-Python-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 🎓 Academic Information

**University:** King Abdulaziz University<br>
**College:** Faculty of Computing and Information Technology<br>
**Department:** Computer Science<br>
**Academic Level:** Masters Degree<br>
**Semester:** Semester 2

## 🎯 Project Overview

This project analyzes the famous Titanic dataset to understand the factors that influenced passenger survival during the disaster. The analysis employs **supervised learning** techniques for **binary classification** to predict whether a passenger would survive or not.

### 🔍 Problem Framing
- **Problem Domain:** Survival prediction analysis
- **Machine Learning Style:** Supervised Learning
- **Model Type:** Binary Classification
- **Target Variable:** Survived (0 = Did not survive, 1 = Survived)

## � Quick Links

- **[� Setup Guide](SETUP_GUIDE.md)** - Detailed installation and usage instructions
- **[📊 Analysis Notebook](notebooks/Titanic_Survival_Analysis_EMAI640_Masters.ipynb)** - Complete Jupyter notebook analysis
- **[📝 Changelog](CHANGELOG.md)** - Project development history
- **[⚙️ Configuration](config/config.py)** - Project settings and parameters

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Installation

1. **Clone or download the project:**
   ```bash
   git clone <......>
   cd .......
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data:**
   - Place your `train.csv` file in the `data/` directory
   - Optionally, place `test.csv` in the same directory

### 3. Running the Analysis

**Option 1: Run the complete analysis pipeline**
```bash
python main.py
```

**Option 2: Use individual modules**
```python
from src.data_processing import TitanicDataProcessor
from src.data_cleaning import TitanicDataCleaner
from src.visualization import TitanicVisualizer

# Load and process data
processor = TitanicDataProcessor()
train_data, test_data = processor.load_data()

# Clean the data
cleaner = TitanicDataCleaner()
cleaned_data = cleaner.clean_dataset(train_data)

# Create visualizations
visualizer = TitanicVisualizer()
visualizer.plot_missing_values_heatmap(train_data)
```

**Option 3: Explore the Jupyter Notebook**
```bash
jupyter notebook notebooks/Titanic_Survival_Analysis_EMAI640_Masters.ipynb
```

## 📊 Analysis Pipeline

The analysis follows a structured pipeline:

### 1. 🔍 Data Loading and Exploration
- Load training and test datasets
- Examine data structure, types, and basic statistics
- Identify numerical and categorical features
- Generate comprehensive data summary

### 2. 🔍 Missing Values Analysis
- Identify missing value patterns
- Calculate missing value percentages
- Create visualizations (heatmaps, bar charts)
- Analyze impact of missing data

### 3. 🧹 Data Cleaning
- **Age:** Group-based median imputation (by Pclass and Sex)
- **Embarked:** Mode imputation
- **Cabin:** Create indicator variable for cabin presence
- **Fare:** Group-based median imputation (by Pclass)

### 4. 📈 Exploratory Data Analysis (EDA)
- Distribution analysis for numerical features
- Frequency analysis for categorical features
- Correlation analysis
- Missing values visualization

### 5. 🎯 Survival Analysis
- Overall survival statistics
- Survival patterns by key features (Sex, Pclass, Embarked)
- Survival rate visualizations

## 📊 Key Features Analyzed

### Numerical Features
- **Pclass:** Passenger class (1st, 2nd, 3rd)
- **Age:** Passenger age in years
- **SibSp:** Number of siblings/spouses aboard
- **Parch:** Number of parents/children aboard
- **Fare:** Ticket fare

### Categorical Features
- **Sex:** Gender (male, female)
- **Embarked:** Port of embarkation (S, C, Q)
- **Name:** Passenger name (used for title extraction)
- **Ticket:** Ticket number
- **Cabin:** Cabin number (high missing rate)

## 🔧 Configuration

The project uses a centralized configuration system in `config/config.py`:

- **Data processing strategies:** Configurable missing value handling
- **Visualization settings:** Plot styles, colors, and formats
- **Model parameters:** Training and evaluation settings
- **File paths:** Centralized path management

## 📈 Key Insights

Based on the analysis, key factors affecting survival include:

1. **Gender:** Females had significantly higher survival rates
2. **Passenger Class:** First-class passengers had better survival chances
3. **Age:** Children and younger passengers had higher survival rates
4. **Family Size:** Optimal family size improved survival chances
5. **Embarkation Port:** Port of embarkation showed survival differences

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas:** Data manipulation and analysis
- **numpy:** Numerical computing
- **matplotlib/seaborn:** Data visualization
- **scikit-learn:** Machine learning utilities
- **jupyter:** Interactive development

## 📝 Usage Examples

### Basic Data Analysis
```python
from src.data_processing import TitanicDataProcessor

# Initialize processor
processor = TitanicDataProcessor('data/train.csv')

# Load and explore data
train_data, _ = processor.load_data()
processor.print_data_summary()

# Get feature types
numerical_features, categorical_features = processor.identify_feature_types()
```

### Data Cleaning
```python
from src.data_cleaning import clean_titanic_data

# Clean data with default strategies
cleaned_data = clean_titanic_data(
    train_data,
    age_strategy='group_median',
    embarked_strategy='mode',
    cabin_strategy='indicator'
)
```

### Visualization
```python
from src.visualization import create_comprehensive_eda_report

# Generate complete EDA report
create_comprehensive_eda_report(
    cleaned_data,
    numerical_features,
    categorical_features
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is created for educational purposes as part of the EMAI 640 course.

---

**Note:** This project demonstrates the complete data science workflow from data exploration to insights generation, following best practices for code organization and documentation.
