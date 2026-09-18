import pandas as pd
import os
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(script_dir, '../../datasets/kidney_disease.csv')
print(f"Loading raw data from {raw_path}...")
df = pd.read_csv(raw_path)

if 'id' in df.columns:
    df = df.drop('id', axis=1)

print("Removing hidden characters...")
df = df.replace(to_replace={'\t': '', '\n': '', ' ': ''}, regex=True) 

cols_to_numeric = ['pcv', 'wc', 'rc']
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Mapping text to numbers...")
mapping = {
    # RBC / PC
    'normal': 1, 'abnormal': 0,
    # PCC / BA
    'present': 1, 'notpresent': 0,
    # HTN / DM / CAD / PE / ANE
    'yes': 1, 'no': 0,
    # APPET
    'good': 1, 'poor': 0,
    # TARGET (Classification)
    'ckd': 1, 'notckd': 0
}

cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
for col in cat_cols:
    df[col] = df[col].map(mapping)

print("Imputing missing values...")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

imputer = SimpleImputer(strategy='median')
df_clean_array = imputer.fit_transform(df)
df_clean = pd.DataFrame(df_clean_array, columns=df.columns)

clean_path = os.path.join(script_dir, '../../datasets/kidney_cleaned.csv')
df_clean.to_csv(clean_path, index=False)
joblib.dump(imputer, os.path.join(script_dir, '../../models/kidney_imputer.pkl'))

print(f"Cleaned data saved to {clean_path}")
print(f"Imputer saved to {os.path.join(script_dir, '../../models/kidney_imputer.pkl')}")