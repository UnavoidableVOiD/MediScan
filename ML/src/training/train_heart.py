import pandas as pd
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../datasets/heart.csv")
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

print("Encoding categorical text data...")

df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})

# B. One-Hot Encoding for Multi-Class columns
# (ChestPainType, RestingECG, ST_Slope)
# drop_first=True avoids the "Dummy Variable Trap"
df_encoded = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True)

print(f"New columns created: {df_encoded.columns.tolist()}")

X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
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


save_path = os.path.join(script_dir, '../../models/heart_best_model.pkl')
scaler_path = os.path.join(script_dir, '../../models/heart_scaler.pkl')
columns_path = os.path.join(script_dir, '../../models/heart_model_columns.pkl')

joblib.dump(best_model_obj, save_path)
joblib.dump(scaler, scaler_path)
joblib.dump(X.columns.tolist(), columns_path)

print("Saved Model, Scaler, and Column Definitions.")