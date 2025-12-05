# Employee Attrition Prediction - Colab-ready script
# Files: employee_attrition_sample_30.csv (upload to Colab / mount drive)
# This script demonstrates end-to-end workflow: load, EDA, preprocess, train, evaluate, save model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

# 1. Load data
df = pd.read_csv('employee_attrition_sample_30.csv')
print('Dataset shape:', df.shape)
print(df.head())

# 2. Basic EDA
print('\nAttrition counts:\n', df['Attrition'].value_counts())
print('\nDepartment distribution:\n', df['Department'].value_counts())

# 3. Preprocessing
# Encode categorical features
cat_cols = ['Department','OverTime','Gender','MaritalStatus']
le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    le_dict[c] = le

# Encode target
le_target = LabelEncoder()
df['AttritionFlag'] = le_target.fit_transform(df['Attrition'])  # Yes=1, No=0

# Features and target
X = df.drop(['emp_id','Attrition','AttritionFlag'], axis=1)
y = df['AttritionFlag']

# Impute (if necessary) and scale
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 5. Model training - Random Forest and Logistic Regression
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('\nRandom Forest Accuracy:', accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('\nLogistic Regression Accuracy:', accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# 6. Feature importance from Random Forest
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print('\nFeature importances:\n', feat_imp)

# 7. ROC-AUC for Random Forest
if len(np.unique(y_test)) > 1:
    y_prob = rf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    print('\nROC AUC (RF):', auc)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr,tpr,label=f'RF AUC={auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(); plt.show()

# 8. Save model and encoders/scaler
joblib.dump(rf, 'rf_attrition_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(imputer, 'imputer.joblib')
joblib.dump(le_dict, 'label_encoders.joblib')
joblib.dump(le_target, 'target_encoder.joblib')

print('\nSaved model and preprocessing artifacts.')
