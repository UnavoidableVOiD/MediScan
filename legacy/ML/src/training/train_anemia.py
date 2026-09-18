import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score 

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../datasets/anemia.csv')

print(f"Loading Anemia Data from {data_path}...")
if not os.path.exists(data_path):
    print(" Error: anemia.csv not found!")
    exit()

df = pd.read_csv(data_path)

target_col = 'Result'
X = df.drop(target_col, axis=1)
y = df[target_col]

imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

if len(X_train) > 20:
    print("⚖️  Applying SMOTE to balance classes...")
    k = min(5, sum(y_train==1)-1, sum(y_train==0)-1)
    if k > 0:
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    else:
        print("Not enough samples for SMOTE. Skipping.")
else:
    print("Dataset too small for SMOTE. Training on raw data.")

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []
best_f1 = 0
best_model_name = ""
best_model_obj = None

print("\n--- MODEL PERFORMANCE ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({"Algorithm": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
    
    if f1 >= best_f1:
        best_f1 = f1
        best_model_name = name
        best_model_obj = model

results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
print(results_df.to_string(index=False))
print(f"\nBest Model: {best_model_name}")

save_dir = os.path.join(script_dir, '../../models')
joblib.dump(best_model_obj, f'{save_dir}/anemia_best_model.pkl')
joblib.dump(scaler, f'{save_dir}/anemia_scaler.pkl')
joblib.dump(imputer, f'{save_dir}/anemia_imputer.pkl')
joblib.dump(X.columns.tolist(), f'{save_dir}/anemia_columns.pkl')

print("Anemia Model Saved.")