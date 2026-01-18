import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../../models")
DATA_DIR = os.path.join(BASE_DIR, "../../datasets")
REPORT_DIR = os.path.join(BASE_DIR, "../../reports/figures")
os.makedirs(REPORT_DIR, exist_ok=True)

def audit_model(disease_name, csv_file, target_col, threshold=0.5):
    print(f"\n==================================================")
    print(f"📊 FINAL AUDIT: {disease_name.upper()}")
    print(f"==================================================")

    # 1. LOAD DATA
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    except FileNotFoundError:
        print(f"❌ Data file not found: {csv_file}")
        return

    # --- CUSTOM PREPROCESSING LOGIC ---
    if disease_name == "diabetes":
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols: df[col] = df[col].replace(0, np.nan)
        
    elif disease_name == "liver":
        if df['Gender'].dtype == 'object':
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        if df['Dataset'].max() == 2:
            df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Drop IDs if present
    drop_cols = [c for c in ['id', 'ID', 'PatientID'] if c in df.columns]
    if drop_cols: 
        X = X.drop(columns=drop_cols)

    # Handle Thyroid string targets
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        y = pd.Series(y) # Convert back to Series for value_counts

    # --- CRITICAL FIX: FILTER RARE CLASSES (For Thyroid) ---
    # We cannot split classes that have < 2 samples.
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    
    if len(valid_classes) < len(class_counts):
        # Filter both X and y
        mask = y.isin(valid_classes)
        X = X[mask]
        y = y[mask]

    # 2. LOAD SAVED ARTIFACTS
    try:
        model = joblib.load(os.path.join(MODELS_DIR, f"{disease_name}_best_model.pkl"))
        imputer = joblib.load(os.path.join(MODELS_DIR, f"{disease_name}_imputer.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, f"{disease_name}_scaler.pkl"))
    except FileNotFoundError:
        print(f"❌ Model artifacts not found for {disease_name}")
        return

    # 3. TRANSFORM
    if disease_name not in ['liver']: 
        X = pd.get_dummies(X, drop_first=True)

    try:
        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)
    except ValueError as e:
        print(f"⚠️ Feature Mismatch Error: {e}")
        return

    # 4. GENERATE t-SNE PLOT
    # print("   -> Generating Final t-SNE Plot...")
    # try:
    #     if len(y) > 1500:
    #         idx = np.random.choice(len(y), 1500, replace=False)
    #         X_vis = X_scaled[idx]
    #         y_vis = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
    #     else:
    #         X_vis = X_scaled
    #         y_vis = y
            
    #     perp = min(30, len(y_vis)-1)
    #     tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    #     X_tsne = tsne.fit_transform(X_vis)

    #     plt.figure(figsize=(8, 6))
    #     sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_vis, palette='coolwarm', alpha=0.8)
    #     plt.title(f"Final t-SNE: {disease_name.title()}")
    #     plt.savefig(os.path.join(REPORT_DIR, f"{disease_name}_final_tsne.png"))
    #     plt.close()
    #     print(f"   -> Plot saved to {disease_name}_final_tsne.png")
    # except Exception as e:
    #     print(f"   -> t-SNE Failed: {e}")

    # 5. VALIDATE PERFORMANCE
    # Split using same seed
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Prediction Logic
    if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
    else:
        preds = model.predict(X_test)

    print("\n🏆 OFFICIAL METRICS:")
    print(f"   Threshold Used: {threshold}")
    
    avg_type = 'weighted' if len(np.unique(y)) > 2 else 'binary'
    
    print(classification_report(y_test, preds))
    rec = recall_score(y_test, preds, average=avg_type, zero_division=0)
    f1 = f1_score(y_test, preds, average=avg_type, zero_division=0)
    
    print(f"   >> RECALL ({avg_type}): {rec:.2%}")
    print(f"   >> F1-SCORE ({avg_type}): {f1:.4f}")

if __name__ == "__main__":
    audit_model("diabetes", "diabetes.csv", "Outcome", threshold=0.30)
    audit_model("liver", "indian_liver_patient.csv", "Dataset", threshold=0.50)
    audit_model("heart", "heart.csv", "HeartDisease")
    audit_model("kidney", "kidney_cleaned.csv", "classification")
    audit_model("anemia", "anemia.csv", "Result")
    audit_model("thyroid", "thyroid_big.csv", "target")