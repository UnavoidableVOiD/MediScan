import joblib
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

class DiseasePredictor:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_path, '../../models')
        self.artifacts = {}
        print(f"Loading AI Models from: {os.path.abspath(self.models_dir)} ...")
        self._load_all_models()

    def _load_all_models(self):
        def load(filename):
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path): return joblib.load(path)
            else: 
                print(f"Warning: Missing {filename}")
                return None

        diseases = ['diabetes', 'heart', 'kidney', 'thyroid', 'anemia', 'liver']
        
        for d in diseases:
            self.artifacts[f'{d}_model'] = load(f'{d}_best_model.pkl')
            self.artifacts[f'{d}_scaler'] = load(f'{d}_scaler.pkl')
            
            if d != 'heart': 
                self.artifacts[f'{d}_imputer'] = load(f'{d}_imputer.pkl')
            
            col_file = 'heart_model_columns.pkl' if d == 'heart' else f'{d}_columns.pkl'
            self.artifacts[f'{d}_cols'] = load(col_file)

        print("All Models Loaded & Hardened.")

    def _prepare_input(self, df, cols_artifact, imputer_artifact, scaler_artifact):
        if cols_artifact:
            df = df.reindex(columns=cols_artifact)

        if imputer_artifact:
            try:
                data_array = imputer_artifact.transform(df)
                cols = cols_artifact if cols_artifact else df.columns
                df = pd.DataFrame(data_array, columns=cols)
            except Exception:
                df = df.fillna(0)
        else:
            df = df.fillna(0)

        if scaler_artifact:
            return scaler_artifact.transform(df)
        return df

    def predict_diabetes(self, data):
        model = self.artifacts.get('diabetes_model')
        if not model: return {"status": "Skipped"}
        try:
            df = pd.DataFrame([data])
            X = self._prepare_input(df, self.artifacts['diabetes_cols'], self.artifacts['diabetes_imputer'], self.artifacts['diabetes_scaler'])
            
            prob = float(model.predict_proba(X)[0][1])
            THRESHOLD = 0.30 
            
            prediction = "Diabetic" if prob >= THRESHOLD else "Healthy"
            
            return {
                "prediction": prediction, 
                "risk_score": round(prob*100, 2),
                "threshold_used": THRESHOLD
            }
        except Exception as e: return {"error": str(e)}

    def predict_kidney(self, data):
        model = self.artifacts.get('kidney_model')
        if not model: return {"status": "Skipped"}
        try:
            df = pd.DataFrame([data])
            X = self._prepare_input(df, self.artifacts['kidney_cols'], self.artifacts['kidney_imputer'], self.artifacts['kidney_scaler'])
            
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1])
            
            return {"prediction": "CKD" if pred==1 else "Healthy", "risk_score": round(prob*100, 2)}
        except Exception as e: return {"error": str(e)}

    def predict_heart(self, data):
        model = self.artifacts.get('heart_model')
        if not model: return {"status": "Skipped"}
        try:
            df = pd.DataFrame([data])
            
            defaults = {
                'Sex': 'M', 'ExerciseAngina': 'N', 'ChestPainType': 'NAP',     
                'RestingECG': 'Normal', 'ST_Slope': 'Up', 'FastingBS': 0,
                'RestingBP': 120, 'MaxHR': 150, 'Oldpeak': 0.0
            }
            for col, val in defaults.items():
                if col not in df.columns: df[col] = val
            
            df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
            df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
            df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'])
            
            X = self._prepare_input(df, self.artifacts['heart_cols'], None, self.artifacts['heart_scaler'])
            
            prob = float(model.predict_proba(X)[0][1])
            THRESHOLD = 0.35
            
            prediction = "Heart Disease" if prob >= THRESHOLD else "Healthy"
            
            return {
                "prediction": prediction, 
                "risk_score": round(prob*100, 2),
                "threshold_used": THRESHOLD
            }
        except Exception as e: return {"error": str(e)}

    def predict_liver(self, data):
        model = self.artifacts.get('liver_model')
        if not model: return {"status": "Skipped"}
        try:
            mapped_data = {
                'Age': data.get('Age', 30),
                'Gender': 1 if 'male' in str(data.get('Gender', '')).lower() else 0,
                'Total_Bilirubin': data.get('Bilirubin_Total'),
                'Direct_Bilirubin': data.get('Direct_Bilirubin'),
                'Alkaline_Phosphotase': data.get('Alkaline_Phosphotase'),
                'Alamine_Aminotransferase': data.get('Alamine_Aminotransferase'),
                'Aspartate_Aminotransferase': data.get('Aspartate_Aminotransferase'),
                'Total_Protiens': data.get('Total_Protiens'),
                'Albumin': data.get('Albumin'),
                'Albumin_and_Globulin_Ratio': data.get('Albumin_and_Globulin_Ratio', 1.0)
            }
            df = pd.DataFrame([mapped_data])
            X = self._prepare_input(df, self.artifacts['liver_cols'], self.artifacts['liver_imputer'], self.artifacts['liver_scaler'])
            
            prob = float(model.predict_proba(X)[0][1])
            THRESHOLD = 0.35
            
            prediction = "Liver Issue" if prob >= THRESHOLD else "Healthy"
            
            return {
                "prediction": prediction, 
                "risk_score": round(prob*100, 2),
                "threshold_used": THRESHOLD
            }
        except Exception as e: return {"error": str(e)}

    def predict_thyroid(self, data):
        model = self.artifacts.get('thyroid_model')
        if not model: return {"status": "Skipped"}
        try:
            df = pd.DataFrame([data])
            X = self._prepare_input(df, self.artifacts['thyroid_cols'], self.artifacts['thyroid_imputer'], self.artifacts['thyroid_scaler'])
            
            pred = model.predict(X)[0] # 0=Normal, 1=Hyper, 2=Hypo
            probs = model.predict_proba(X)[0]
            
            labels = {0: "Normal", 1: "Hyperthyroid", 2: "Hypothyroid"}
            prediction_label = labels.get(pred, "Unknown")
            
            if pred == 0:
                risk_score = (1 - probs[0]) * 100
            else:
                risk_score = probs[pred] * 100
            
            return {
                "prediction": prediction_label, 
                "risk_score": round(risk_score, 2)
            }
        except Exception as e: return {"error": str(e)}

    def predict_anemia(self, data):
        model = self.artifacts.get('anemia_model')
        if not model: return {"status": "Skipped"}
        try:
            df = pd.DataFrame([data])
            if 'Gender' in df.columns:
                val = str(df['Gender'].iloc[0]).lower()
                if val in ['male', 'm', '1']: df['Gender'] = 1
                else: df['Gender'] = 0
            
            X = self._prepare_input(df, self.artifacts['anemia_cols'], self.artifacts['anemia_imputer'], self.artifacts['anemia_scaler'])
            
            pred = int(model.predict(X)[0])
            
            return {"prediction": "Anemia Detected" if pred==1 else "Healthy", "status": "Abnormal" if pred==1 else "Normal"}
        except Exception as e: return {"error": str(e)}

    def analyze_full_report(self, extracted_data):
        report = {}
        if 'Glucose' in extracted_data: report['Diabetes'] = self.predict_diabetes(extracted_data)
        if 'Creatinine' in extracted_data or 'Blood_Urea' in extracted_data: report['Kidney'] = self.predict_kidney(extracted_data)
        if 'Bilirubin_Total' in extracted_data: report['Liver'] = self.predict_liver(extracted_data)
        if 'TSH' in extracted_data: report['Thyroid'] = self.predict_thyroid(extracted_data)
        if 'Hemoglobin' in extracted_data: report['Anemia'] = self.predict_anemia(extracted_data)
        if 'Cholesterol' in extracted_data: report['Heart'] = self.predict_heart(extracted_data)
        return report

if __name__ == "__main__":
    engine = DiseasePredictor()
    
    print("\n--- RUNNING COMPLETE SYSTEM CHECK ---")
    
    mock_data = {
    "Alamine_Aminotransferase": 4.0,
    "Blood_Urea": 25.0,
    "Creatinine": 9.0,
    "Sodium": 138.2,
    "Potassium": 44.0,
    "TSH": 2.51,
    "MCV": 91.9,
    "MCH": 32.8,
    "MCHC": 35.7
}
    
    import json
    print(json.dumps(engine.analyze_full_report(mock_data), indent=4))