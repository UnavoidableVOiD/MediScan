import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../datasets/kidney_cleaned.csv')
models_dir = os.path.join(script_dir, '../../models')
os.makedirs(models_dir, exist_ok=True)

print(f"Loading clean data from {data_path}...")
df = pd.read_csv(data_path)

target_col = 'classification'

REAL_WORLD_FEATURES = [
    'sc',    # Serum Creatinine 
    'bu',    # Blood Urea
    'hemo',  # Hemoglobin
    
    'sod',   # Sodium
    'pot',   # Potassium
    
    'sg',    # Specific Gravity
    'al',    # Albumin
    'su',    # Sugar

    'age',   # Age
    'bp',    # Blood Pressure
    'bgr',   # Random Blood Glucose
    'htn',   # Hypertension History (Yes/No)
    'dm'     # Diabetes History (Yes/No)
]

available_cols = [col for col in REAL_WORLD_FEATURES if col in df.columns]
print(f"\nTraining restricted to {len(available_cols)} Real-World Features: {available_cols}")

X = df[available_cols]
y = df[target_col]

print("Fitting Imputer (Median)...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

print("Scaling Data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True)
}

results = []
best_f1 = 0
best_model_name = ""
best_model_obj = None

print("\n--- TRAINING BENCHMARK ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    results.append({"Model": name, "F1": f1})
    print(f"   -> {name}: F1 Score = {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model_obj = model

print(f"\nWINNER: {best_model_name} (F1: {best_f1:.2f})")
print("\nDetailed Report:")
print(classification_report(y_test, best_model_obj.predict(X_test)))

print("Saving artifacts...")
joblib.dump(best_model_obj, os.path.join(models_dir, 'kidney_best_model.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'kidney_scaler.pkl'))
joblib.dump(imputer, os.path.join(models_dir, 'kidney_imputer.pkl'))
joblib.dump(available_cols, os.path.join(models_dir, 'kidney_columns.pkl'))

print("Kidney Model Retrained & Saved Successfully.")