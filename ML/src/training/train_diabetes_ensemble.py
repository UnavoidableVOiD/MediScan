import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../datasets/diabetes.csv")
df = pd.read_csv(data_path)

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols: df[col] = df[col].replace(0, np.nan)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

clf1 = XGBClassifier(scale_pos_weight=2, use_label_encoder=False, eval_metric='logloss', random_state=42)

clf2 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

clf3 = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)

# voting='soft' means we average the probabilities (e.g., 80% + 40% + 60% = 60% avg)
eclf = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2), ('svm', clf3)], voting='soft')

print("🗳️ Training Ensemble Model...")
eclf.fit(X_train_res, y_train_res)

y_proba = eclf.predict_proba(X_test)[:, 1]
y_pred_optimized = (y_proba >= 0.35).astype(int)

print("\nENSEMBLE RESULTS")
print(f"Recall:    {recall_score(y_test, y_pred_optimized):.4f} (Goal: >0.90)")
print(f"Precision: {precision_score(y_test, y_pred_optimized):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_optimized):.4f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_optimized):.4f}")

save_dir = os.path.join(script_dir, '../../models')
joblib.dump(eclf, f'{save_dir}/diabetes_best_model.pkl')
joblib.dump(scaler, f'{save_dir}/diabetes_scaler.pkl')
joblib.dump(imputer, f'{save_dir}/diabetes_imputer.pkl')
joblib.dump(X.columns.tolist(), f'{save_dir}/diabetes_model_columns.pkl')
print("Saved Diabetes Ensemble.")