import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, f1_score

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../../models")
FIGURES_DIR = os.path.join(BASE_DIR, "../../reports/figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def train_disease_model(disease_name, csv_path, target_col, selected_features=None):
    print(f"STARTING ANALYSIS FOR: {disease_name.upper()}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return

    if target_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == target_col.lower()]
        if matches:
            target_col = matches[0]
            print(f"Found case-insensitive match for target: '{target_col}'")
        else:
            print(f"Error: Target column '{target_col}' not found!")
            print(f"Available columns: {df.columns.tolist()}")
            guessed = df.columns[-1]
            print(f"Switching to guessed target: '{guessed}'")
            target_col = guessed

    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
    
    drop_cols = [c for c in ['id', 'ID', 'PatientID'] if c in df.columns]
    if drop_cols: df = df.drop(columns=drop_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if selected_features:
        valid_feats = [f for f in selected_features if f in X.columns]
        X = X[valid_feats]

    X = pd.get_dummies(X, drop_first=True) 
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    feature_names = X.columns.tolist()

    print(f"Generating t-SNE plot for report...")
    try:
        perp = min(30, len(y)-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        X_tsne = tsne.fit_transform(X_scaled)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='coolwarm', alpha=0.7)
        plt.title(f"t-SNE Clustering: {disease_name}")
        plt.savefig(os.path.join(FIGURES_DIR, f"{disease_name}_tsne.png"))
        plt.close()
    except Exception as e:
        print(f"   -> t-SNE Failed: {e}")

    # 4. SPLIT
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index
    
    if len(rare_classes) > 0:
        print(f"Warning: Dropping rare classes with < 2 samples: {rare_classes.tolist()}")
        mask = y.isin(class_counts[class_counts >= 2].index)
        X_scaled = X_scaled[mask]
        y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Running GridSearchCV (Optimizing F1-Score)...")
    
    models_to_test = [
        {
            "name": "SVM",
            "estimator": SVC(class_weight='balanced', probability=True),
            "params": {
                'C': [1, 10, 100],
                'gamma': ['scale', 0.1],
                'kernel': ['rbf'] 
            }
        },
        {
            "name": "RandomForest",
            "estimator": RandomForestClassifier(class_weight='balanced', random_state=42),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [10, 20]
            }
        }
    ]

    best_model_overall = None
    best_f1 = -1 
    best_name = ""

    for m in models_to_test:
        grid = GridSearchCV(m['estimator'], m['params'], scoring='f1_weighted', cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        
        y_pred = grid.best_estimator_.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, average='weighted') 
        rec = recall_score(y_test, y_pred, average='weighted')
        
        print(f"   -> {m['name']} | F1: {f1:.4f} | Recall: {rec:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_overall = grid.best_estimator_
            best_name = m['name']

    print(f"\n🏆 WINNER: {best_name} (F1: {best_f1:.2f})")
    print(classification_report(y_test, best_model_overall.predict(X_test)))
    
    # 6. SAVE ARTIFACTS
    joblib.dump(best_model_overall, os.path.join(MODELS_DIR, f"{disease_name}_best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{disease_name}_scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODELS_DIR, f"{disease_name}_imputer.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, f"{disease_name}_columns.pkl"))
    print(f"Saved artifacts for {disease_name}.\n")


if __name__ == "__main__":
    
    # 1. DIABETES
    train_disease_model(
        "diabetes", 
        os.path.join(BASE_DIR, "../../datasets/diabetes.csv"), 
        target_col="Outcome" 
    )

    # 2. HEART
    train_disease_model(
        "heart", 
        os.path.join(BASE_DIR, "../../datasets/heart.csv"), 
        target_col="HeartDisease"
    )

    # 3. LIVER
    train_disease_model(
        "liver", 
        os.path.join(BASE_DIR, "../../datasets/indian_liver_patient.csv"), 
        target_col="Dataset" 
    )
    
    # 4. THYROID
    train_disease_model(
        "thyroid",
        os.path.join(BASE_DIR, "../../datasets/thyroid_big.csv"),
        target_col="target" 
    )

    # 5. KIDNEY
    train_disease_model(
        "kidney",
        os.path.join(BASE_DIR, "../../datasets/kidney_cleaned.csv"),
        target_col="classification" 
    )

    # 6. ANEMIA
    train_disease_model(
        "anemia",
        os.path.join(BASE_DIR, "../../datasets/anemia.csv"),
        target_col="Result" 
    )