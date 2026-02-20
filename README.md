
# Employee Attrition Prediction  

This project predicts employee attrition using machine learning.  
(Add your full project description here)
# 📊 Employee Attrition Prediction - Colab-ready Script

This project demonstrates an end-to-end workflow for predicting employee attrition using a sample dataset (`employee_attrition_sample_30.csv`). The workflow includes loading data, exploratory data analysis (EDA), preprocessing, model training, evaluation, and saving trained artifacts.

---

## Features

* Load and inspect employee attrition dataset
* Exploratory Data Analysis (EDA)
* Preprocessing: encoding categorical variables, scaling, and imputation
* Model training: Random Forest and Logistic Regression
* Evaluate model: accuracy, classification report, feature importance, ROC-AUC
* Save trained model and preprocessing artifacts for later use

---

## Dataset

* File: `employee_attrition_sample_30.csv` (upload to Colab or mount from Google Drive)
* Target column: `Attrition` (Yes/No)
* Sample columns: `Department`, `OverTime`, `Gender`, `MaritalStatus`, `emp_id`, and other numerical features

---

## Installation

1. **Clone or download the script to Colab or local machine**

2. **Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## Usage

### 1. Load Dataset

```python
import pandas as pd
df = pd.read_csv('employee_attrition_sample_30.csv')
print(df.head())
```

### 2. Exploratory Data Analysis (EDA)

* Inspect attrition counts, department distribution, etc.
* Visualizations using `matplotlib` or `seaborn`

### 3. Preprocessing

* Encode categorical features (`Department`, `OverTime`, `Gender`, `MaritalStatus`) using `LabelEncoder`
* Encode target `Attrition` as `AttritionFlag` (Yes=1, No=0)
* Impute missing values with `SimpleImputer(strategy='median')`
* Scale features with `StandardScaler`

### 4. Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
```

### 5. Model Training

* **Random Forest**

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
```

* **Logistic Regression**

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
```

### 6. Model Evaluation

* Accuracy, classification report, confusion matrix
* Feature importance from Random Forest
* ROC-AUC and ROC curve

### 7. Save Model & Artifacts

```python
import joblib
joblib.dump(rf, 'rf_attrition_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(imputer, 'imputer.joblib')
joblib.dump(le_dict, 'label_encoders.joblib')
joblib.dump(le_target, 'target_encoder.joblib')
```

---

## Notes

* This script is ready to run in Google Colab; ensure dataset is uploaded or mounted.
* Random Forest usually performs better for classification tasks with tabular data.
* All preprocessing artifacts (scaler, imputer, encoders) are saved for consistent inference on new data.

---

## License

MIT License
