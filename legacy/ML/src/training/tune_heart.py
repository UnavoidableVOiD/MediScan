import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../datasets/heart.csv")
df = pd.read_csv(data_path)

df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True)

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=5, 
    use_label_encoder=False, 
    eval_metric='logloss'
)

print("Training Heart Model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nHEART RESULTS")
print(f"Recall:    {recall_score(y_test, y_pred):.4f} (Goal: >0.95)")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")  
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")

save_dir = os.path.join(script_dir, '../../models')
joblib.dump(model, f'{save_dir}/heart_best_model.pkl')
joblib.dump(scaler, f'{save_dir}/heart_scaler.pkl')
joblib.dump(X.columns.tolist(), f'{save_dir}/heart_model_columns.pkl')
print("Saved Heart Model.")