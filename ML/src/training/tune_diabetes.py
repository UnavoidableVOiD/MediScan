import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../datasets/diabetes.csv")
df = pd.read_csv(data_path)

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols: df[col] = df[col].replace(0, np.nan)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_processed = imputer.fit_transform(X_train)
X_test_processed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

param_grid = {
    'scale_pos_weight': [5, 8, 12], 
    'max_depth': [3, 4],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 200]
}

xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)

print(f"{'='*50}")
print("STARTING RECALL OPTIMIZATION")
print(f"{'='*50}")

grid = GridSearchCV(xgb, param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
print(f"\nBest Aggression Config: {grid.best_params_}")

y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

THRESHOLD = 0.30 
y_pred_optimized = (y_proba >= THRESHOLD).astype(int)

rec = recall_score(y_test, y_pred_optimized)
prec = precision_score(y_test, y_pred_optimized)

print(f"\nFINAL RESULTS (Threshold {THRESHOLD}):")
print(f"Recall:    {rec:.2%}  <-- THIS SHOULD BE >90%")
print(f"Precision: {prec:.2%}")

cm = confusion_matrix(y_test, y_pred_optimized)
print(f"\nMissed Cases (False Negatives): {cm[1][0]} (Should be low!)")

save_dir = os.path.join(script_dir, '../../models')
joblib.dump(best_model, f'{save_dir}/diabetes_best_model.pkl')
joblib.dump(scaler, f'{save_dir}/diabetes_scaler.pkl')
joblib.dump(imputer, f'{save_dir}/diabetes_imputer.pkl')
joblib.dump(X.columns.tolist(), f'{save_dir}/diabetes_columns.pkl')

print("\nSaved Aggressive Model to 'models/' folder.")