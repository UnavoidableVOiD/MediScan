import pandas as pd
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
from xgboost import XGBClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../datasets/kidney_cleaned.csv')
print(f"Loading clean data from {data_path}...")
df = pd.read_csv(data_path)

target_col = 'classification'
X = df.drop(target_col, axis=1)
y = df[target_col]

print("Fitting Imputer...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []

best_f1 = 0
best_model_name = ""
best_model_obj = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Algorithm": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model_obj = model

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-Score", ascending=False)
print(results_df.to_string(index=False))

print(f"\nBest Model: {best_model_name} (F1: {best_f1:.2f})")

save_path = os.path.join(script_dir,'../../models/kidney_best_model.pkl')
scalar_path = os.path.join(script_dir,'../../models/kidney_scaler.pkl')
imputer_path = os.path.join(script_dir,'../../models/kidney_imputer.pkl') # <--- New
columns_path = os.path.join(script_dir,'../../models/kidney_columns.pkl') # <--- New

joblib.dump(best_model_obj, save_path)
joblib.dump(scaler, scalar_path)
joblib.dump(imputer, imputer_path)
joblib.dump(X.columns.tolist(), columns_path) 

print("Saved Model, Scaler, Imputer, and Columns.")