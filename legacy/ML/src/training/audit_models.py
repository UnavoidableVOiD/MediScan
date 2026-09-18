import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../../models")
DATA_DIR = os.path.join(BASE_DIR, "../../datasets")

def load_artifact(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

def audit_diabetes():
    print("\n" + "="*50)
    print("📊 FINAL AUDIT: DIABETES")
    print("="*50)
    
    df = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
    
    # Same pre-processing as training
    cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for c in cols_zero:
        df[c] = df[c].replace(0, np.nan)
        
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    model = load_artifact("diabetes_best_model.pkl")
    scaler = load_artifact("diabetes_scaler.pkl")
    imputer = load_artifact("diabetes_imputer.pkl")
    
    if not model: return
    
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    
    # Diabetes uses probability thresholding
    probs = model.predict_proba(X_scaled)[:, 1]
    THRESHOLD = 0.3
    y_pred = (probs >= THRESHOLD).astype(int)
    
    print(f"\n🏆 OFFICIAL METRICS:")
    print(f"   Threshold Used: {THRESHOLD}")
    print(classification_report(y, y_pred))
    print(f"   >> RECALL (binary): {recall_score(y, y_pred)*100:.2f}%")
    print(f"   >> F1-SCORE (binary): {f1_score(y, y_pred)*100:.4f}")

def audit_heart():
    print("\n" + "="*50)
    print("📊 FINAL AUDIT: HEART")
    print("="*50)
    
    df = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))
    
    # Encoding (Same as inference logic)
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'])
    
    # Align columns
    saved_cols = load_artifact("heart_model_columns.pkl")
    if saved_cols is None: saved_cols = load_artifact("heart_columns.pkl")
    
    # Reindex to match model
    df = df.reindex(columns=saved_cols, fill_value=0)
    
    # Heart CSV has 'HeartDisease' target, but reindex might have dropped it or it's not in X
    # We need to reload y from raw CSV
    y = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))["HeartDisease"]
    
    model = load_artifact("heart_best_model.pkl")
    scaler = load_artifact("heart_scaler.pkl")
    
    X_scaled = scaler.transform(df)
    
    probs = model.predict_proba(X_scaled)[:, 1]
    THRESHOLD = 0.5
    y_pred = (probs >= THRESHOLD).astype(int)
    
    print(f"\n🏆 OFFICIAL METRICS:")
    print(f"   Threshold Used: {THRESHOLD}")
    print(classification_report(y, y_pred))
    print(f"   >> RECALL (binary): {recall_score(y, y_pred)*100:.2f}%")
    print(f"   >> F1-SCORE (binary): {f1_score(y, y_pred)*100:.4f}")

def audit_kidney():
    print("\n" + "="*50)
    print("📊 FINAL AUDIT: KIDNEY")
    print("="*50)
    
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "kidney_cleaned.csv"))
        y = df['classification']
        
        # --- FIX: Load the specific columns the model was trained on ---
        trained_cols = load_artifact("kidney_columns.pkl")
        
        if trained_cols:
            # Filter X to only the 13 real-world columns
            X = df[trained_cols]
        else:
            X = df.drop('classification', axis=1) # Fallback (will crash if mismatch)

        model = load_artifact("kidney_best_model.pkl")
        imputer = load_artifact("kidney_imputer.pkl")
        scaler = load_artifact("kidney_scaler.pkl")
        
        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)
        
        y_pred = model.predict(X_scaled)
        
        print(f"\n🏆 OFFICIAL METRICS:")
        print(classification_report(y, y_pred))
        print(f"   >> RECALL (binary): {recall_score(y, y_pred)*100:.2f}%")
        print(f"   >> F1-SCORE (binary): {f1_score(y, y_pred)*100:.4f}")
        
    except Exception as e:
        print(f"⚠️ Error: {e}")

def audit_liver():
    print("\n" + "="*50)
    print("📊 FINAL AUDIT: LIVER")
    print("="*50)
    
    df = pd.read_csv(os.path.join(DATA_DIR, "indian_liver_patient.csv"))
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
    
    X = df.drop("Dataset", axis=1)
    y = df["Dataset"]
    
    model = load_artifact("liver_best_model.pkl")
    imputer = load_artifact("liver_imputer.pkl")
    scaler = load_artifact("liver_scaler.pkl")
    
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    
    probs = model.predict_proba(X_scaled)[:, 1]
    THRESHOLD = 0.5
    y_pred = (probs >= THRESHOLD).astype(int)
    
    print(f"\n🏆 OFFICIAL METRICS:")
    print(f"   Threshold Used: {THRESHOLD}")
    print(classification_report(y, y_pred))
    print(f"   >> RECALL (binary): {recall_score(y, y_pred)*100:.2f}%")
    print(f"   >> F1-SCORE (binary): {f1_score(y, y_pred)*100:.4f}")

def audit_anemia():
    print("\n" + "="*50)
    print("📊 FINAL AUDIT: ANEMIA")
    print("="*50)
    
    df = pd.read_csv(os.path.join(DATA_DIR, "anemia.csv"))
    X = df.drop("Result", axis=1)
    y = df["Result"]
    
    model = load_artifact("anemia_best_model.pkl")
    imputer = load_artifact("anemia_imputer.pkl")
    scaler = load_artifact("anemia_scaler.pkl")
    
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    y_pred = model.predict(X_scaled)
    
    print(f"\n🏆 OFFICIAL METRICS:")
    print(classification_report(y, y_pred))
    print(f"   >> RECALL (binary): {recall_score(y, y_pred)*100:.2f}%")
    print(f"   >> F1-SCORE (binary): {f1_score(y, y_pred)*100:.4f}")

def audit_thyroid():
    print("\n" + "="*50)
    print("📊 FINAL AUDIT: THYROID")
    print("="*50)
    
    df = pd.read_csv(os.path.join(DATA_DIR, "thyroid_big.csv"))
    
    # Preprocessing (Label Encode target)
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'].astype(str))
    
    drop_cols = [c for c in ['id', 'ID', 'PatientID'] if c in df.columns]
    df = df.drop(columns=drop_cols)
    
    X = df.drop("target", axis=1)
    y = df["target"]
    X = pd.get_dummies(X, drop_first=True)
    
    model = load_artifact("thyroid_best_model.pkl")
    imputer = load_artifact("thyroid_imputer.pkl")
    scaler = load_artifact("thyroid_scaler.pkl")
    
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    y_pred = model.predict(X_scaled)
    
    print(f"\n🏆 OFFICIAL METRICS:")
    print(classification_report(y, y_pred))
    print(f"   >> RECALL (weighted): {recall_score(y, y_pred, average='weighted')*100:.2f}%")
    print(f"   >> F1-SCORE (weighted): {f1_score(y, y_pred, average='weighted')*100:.4f}")

if __name__ == "__main__":
    audit_diabetes()
    audit_liver()
    audit_heart()
    audit_kidney()
    audit_anemia()
    audit_thyroid()