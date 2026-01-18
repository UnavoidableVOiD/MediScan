import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../../models")
FIGURES_DIR = os.path.join(BASE_DIR, "../../reports/figures")
DATA_DIR = os.path.join(BASE_DIR, "../../datasets")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def generate_tsne(X, y, name):
    print(f"   -> Generating Refined t-SNE for {name}...")
    try:
        # Downsample if too large for speed
        if len(y) > 2000:
            idx = np.random.choice(len(y), 2000, replace=False)
            X_vis = X[idx]
            y_vis = y.iloc[idx]
        else:
            X_vis = X
            y_vis = y

        perp = min(30, len(y_vis)-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        X_tsne = tsne.fit_transform(X_vis)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_vis, palette='coolwarm', alpha=0.7)
        plt.title(f"Refined t-SNE: {name}")
        plt.savefig(os.path.join(FIGURES_DIR, f"{name}_refined_tsne.png"))
        plt.close()
    except Exception as e:
        print(f"   -> t-SNE skipped: {e}")

def train_diabetes_v2():
    print("REFINING DIABETES (Median Imputation)")
    
    df = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
    
    # 0 is invalid for these columns we treat 0 as NaN, then the Imputer will fill it with the Median.
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Count zeros before
    zeros_before = (df[cols_with_zeros] == 0).sum().sum()
    print(f"   -> Found {zeros_before} invalid '0' values. Marking as missing...")
    
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    #IMPUTATION for MEDIAN
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    
    feature_names = X.columns.tolist()

    generate_tsne(X_scaled, y, "diabetes")

    #MODEL: VOTING ENSEMBLE
    # Combining models helps when the decision boundary is fuzzy (like in Diabetes)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    clf1 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf2 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=2)
    clf3 = SVC(probability=True, class_weight='balanced', kernel='rbf')

    voting_clf = VotingClassifier(
        estimators=[('rf', clf1), ('xgb', clf2), ('svm', clf3)],
        voting='soft'
    )
    
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    
    print("\n🏆 DIABETES V2 RESULTS:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(voting_clf, os.path.join(MODELS_DIR, "diabetes_best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "diabetes_scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODELS_DIR, "diabetes_imputer.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "diabetes_columns.pkl"))

def train_liver_v2():
    print("REFINING LIVER (Voting Ensemble)")
    
    df = pd.read_csv(os.path.join(DATA_DIR, "indian_liver_patient.csv"))
    
    #CLEANING
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0}) # 1=Sick, 0=Healthy
    
    X = df.drop("Dataset", axis=1)
    y = df["Dataset"]
    
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    feature_names = X.columns.tolist()

    generate_tsne(X_scaled, y, "liver")

    #VOTING CLASSIFIER
    # Liver is noisy. Logistic Regression acts as a stabilizer.
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    clf1 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf2 = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf3 = SVC(probability=True, class_weight='balanced', kernel='rbf')

    voting_clf = VotingClassifier(
        estimators=[('rf', clf1), ('lr', clf2), ('svm', clf3)],
        voting='soft'
    )
    
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    
    print("\n🏆 LIVER V2 RESULTS:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(voting_clf, os.path.join(MODELS_DIR, "liver_best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "liver_scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODELS_DIR, "liver_imputer.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "liver_columns.pkl"))

if __name__ == "__main__":
    train_diabetes_v2()
    train_liver_v2()