import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. Load
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir,'../../datasets/thyroid_big.csv')
print(f"Loading Big Thyroid Data...")
df = pd.read_csv(data_path, na_values=[''])

# 2. Clean & Map Targets
cols_to_drop = ['patient_id', 'referral_source', 'TBG'] 
cols_to_drop += [c for c in df.columns if 'measured' in c]
df = df.drop(columns=cols_to_drop, errors='ignore')

def map_target(val):
    val = str(val).lower().strip()
    if 'hyper' in val or val in ['a', 'b', 'c', 'd']: return 1
    elif 'hypo' in val or val in ['e', 'f', 'g', 'h']: return 2
    return 0

df['Target_Class'] = df['target'].apply(map_target)
df = df.drop(columns=['target'])

# Map Sex/Boolean
for col in df.select_dtypes(include=['object']).columns:
    if col == 'sex': df[col] = df[col].map({'M': 1, 'F': 0})
    else: df[col] = df[col].map({'t': 1, 'f': 0})

# Convert numeric
for col in ['TSH', 'T3', 'TT4', 'T4U', 'FTI']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Split X and y (CRITICAL STEP)
X = df.drop('Target_Class', axis=1) # <--- Target removed from X
y = df['Target_Class']

# 4. Impute & Scale
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# 5. Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 6. Save Everything (Including Columns)
save_dir = os.path.join(script_dir, '../../models')
joblib.dump(model, f'{save_dir}/thyroid_best_model.pkl')
joblib.dump(scaler, f'{save_dir}/thyroid_scaler.pkl')
joblib.dump(imputer, f'{save_dir}/thyroid_imputer.pkl')
joblib.dump(X.columns.tolist(), f'{save_dir}/thyroid_columns.pkl')

print("Thyroid Model Retrained & Saved (Clean).")