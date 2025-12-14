import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../datasets/indian_liver_patient.csv")
print(f"Loading Liver Data from {data_path}...")
df = pd.read_csv(data_path)

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

X = df.drop('Dataset', axis=1)
y = df['Dataset']

print(f"Features: {X.shape[1]} columns")
if 'Dataset' in X.columns:
    raise ValueError("CRITICAL: Target is still in features!")

imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Neural Network": MLPClassifier(max_iter=500)
}

results = []
best_score = 0 
best_model_name = ""
best_model_obj = None

print("\n--- COMPARING ALGORITHMS (Honest Test) ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Algorithm": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })
    
    if f1 > best_score:
        best_score = f1
        best_model_name = name
        best_model_obj = model

results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print(results_df.to_string(index=False))
print(f"\nBest Model: {best_model_name} (F1: {best_score:.2f})")

save_dir = os.path.join(script_dir, '../../models')
joblib.dump(best_model_obj, f'{save_dir}/liver_best_model.pkl')
joblib.dump(scaler, f'{save_dir}/liver_scaler.pkl')
joblib.dump(imputer, f'{save_dir}/liver_imputer.pkl')
joblib.dump(X.columns.tolist(), f'{save_dir}/liver_columns.pkl') # Saving the clean column list

print("Liver Model Retrained & Saved.")