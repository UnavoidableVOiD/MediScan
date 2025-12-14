import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.metrics import recall_score, precision_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../../models')
DATA_DIR = os.path.join(BASE_DIR, '../../datasets')

def load_artifacts(disease):
    try:
        model = joblib.load(os.path.join(MODELS_DIR, f'{disease}_best_model.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, f'{disease}_scaler.pkl'))
        
        col_file = f'{disease}_model_columns.pkl' if disease == 'heart' else f'{disease}_columns.pkl'
        cols = joblib.load(os.path.join(MODELS_DIR, col_file))
        
        try:
            imputer = joblib.load(os.path.join(MODELS_DIR, f'{disease}_imputer.pkl'))
        except:
            imputer = None
            
        return model, scaler, imputer, cols
    except Exception as e:
        print(f"Could not load artifacts for {disease}: {e}")
        return None, None, None, None

def audit_disease(disease, csv_name, target_col):
    print(f"\n{'='*50}")
    print(f"AUDITING: {disease.upper()}")
    print(f"{'='*50}")

    try:
        df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    except:
        print(f"CSV not found: {csv_name}")
        return

    X = df.drop(columns=[target_col], errors='ignore')
    y_raw = df[target_col]

    if disease == 'heart':
        if 'Sex' in X.columns: X['Sex'] = X['Sex'].map({'M': 1, 'F': 0})
        if 'ExerciseAngina' in X.columns: X['ExerciseAngina'] = X['ExerciseAngina'].map({'Y': 1, 'N': 0})
        X = pd.get_dummies(X, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True)
        y = y_raw

    elif disease == 'thyroid':
        def map_target(val):
            val = str(val).lower()
            if 'hyper' in val or 'a' in val or 'b' in val: return 1
            if 'hypo' in val or 'e' in val or 'f' in val: return 1 
            return 0
        y = y_raw.apply(map_target)

        cols_to_drop = ['patient_id', 'referral_source', 'TBG'] 
        cols_to_drop += [c for c in df.columns if 'measured' in c]
        X = X.drop(columns=cols_to_drop, errors='ignore')

        for col in X.select_dtypes(include=['object']).columns:
            if col == 'sex': X[col] = X[col].map({'M': 1, 'F': 0})
            else: X[col] = X[col].map({'t': 1, 'f': 0, 'y': 1, 'n': 0})

    elif disease == 'liver':
        y = y_raw.map({1: 1, 2: 0})
        if 'Gender' in X.columns:
            X['Gender'] = X['Gender'].apply(lambda x: 1 if str(x).strip() == 'Male' else 0)

    elif disease == 'anemia':
        if 'Gender' in X.columns:
             X['Gender'] = X['Gender'].apply(lambda x: 1 if str(x).lower() in ['m', 'male', '1'] else 0)
        y = y_raw 

    else: 
        y = y_raw

    model, scaler, imputer, model_cols = load_artifacts(disease)
    if not model: return

    X = X.reindex(columns=model_cols, fill_value=0)

    if imputer: X_processed = imputer.transform(X)
    else: X_processed = X
    
    if scaler: X_processed = scaler.transform(X_processed)
    
    print("\nTOP 3 FEATURES (Logic Check):")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(3, len(indices))):
            print(f"   {i+1}. {model_cols[indices[i]]}: {importances[indices[i]]:.4f}")
    elif hasattr(model, 'coef_'): 
        importances = np.abs(model.coef_[0])
        indices = np.argsort(importances)[::-1]
        for i in range(min(3, len(indices))):
            print(f"   {i+1}. {model_cols[indices[i]]}: {importances[indices[i]]:.4f}")
    else:
        print("   (Model does not expose feature importance)")

    print("\nTHRESHOLD ANALYSIS (Recall/Safety):")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_processed)
        
        if probs.shape[1] > 2:
            pos_probs = 1 - probs[:, 0] 
        else:
            pos_probs = probs[:, 1]

        print(f"   {'Threshold':<10} | {'Recall':<10} | {'Precision':<10} | {'Action'}")
        print(f"   {'-'*55}")
        
        for t in [0.3, 0.4, 0.5]:
            y = y.fillna(0)
            preds = (pos_probs >= t).astype(int)
            
            rec = recall_score(y, preds, average='binary', pos_label=1, zero_division=0)
            prec = precision_score(y, preds, average='binary', pos_label=1, zero_division=0)
            
            marker = "✅" if t == 0.3 else ""
            print(f"   {t:<10} | {rec:.2%}     | {prec:.2%}     | {marker}")

if __name__ == "__main__":
    audit_disease('diabetes', 'diabetes.csv', 'Outcome')
    audit_disease('heart', 'heart.csv', 'HeartDisease')
    audit_disease('kidney', 'kidney_cleaned.csv', 'classification') 
    audit_disease('liver', 'indian_liver_patient.csv', 'Dataset')
    audit_disease('thyroid', 'thyroid_big.csv', 'target')
    audit_disease('anemia', 'anemia.csv', 'Result') 