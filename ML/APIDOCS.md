# 📘 MediScan AI API Documentation

**Base URL:** `http://localhost:8001`

---

## 1. Analyze Lab Report (OCR + AI Prediction)
This is the core endpoint for patients. It uploads a PDF medical report, extracts biochemical data using OCR, checks for critical values (Safety Guard), and runs diagnostic ML models.

- **Endpoint:** `POST /analyze_report`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

### Request Body
| Parameter | Type | Required | Description |
| :--- | :---: | :---: | :--- |
| `file` | File (PDF) | Yes | The medical lab report file to be analyzed. |

### Success Response (JSON)
```json
{
  "source": "ocr_extraction",
  "tier": "Free User",
  "raw_data": {
    "Glucose": 160.5,
    "Creatinine": 1.2,
    "Hemoglobin": 13.8,
    "Potassium": 7.0
  },
  "critical_alerts": [
    "CRITICAL: Potassium is 7.0 (Immediate Cardiac Risk)"
  ],
  "health_analysis": {
    "Diabetes": {
      "prediction": "Diabetic",
      "risk_score": 88.5,
      "threshold_used": 0.3
    },
    "Kidney": {
      "prediction": "Healthy",
      "risk_score": 12.0
    }
  },
  "summary_for_patient": "URGENT: Your potassium is critically high. Please see a doctor immediately...",
  "summary_for_doctor": "Clinical Impression: Hyperkalemia detected. Correlate with ECG..."
}
```

---

## 2. MediBot AI Chat (RAG)
An interactive AI assistant that provides medical insights based on the patient's analyzed report data.

- **Endpoint:** `POST /chat`
- **Method:** `POST`
- **Content-Type:** `application/json`

### Request Body (JSON)
```json
{
  "question": "What foods should I avoid given my glucose level?",
  "patient_context": {
    "Glucose": 160.5,
    "Hemoglobin": 13.8
  },
  "is_premium": true
}
```

### Success Response
```json
{
  "question": "What foods should I avoid given my glucose level?",
  "answer": "Since your glucose is elevated (160.5 mg/dL), you should avoid processed sugars...",
  "tier": "Premium User"
}
```

---

## 3. B2B Bulk Analysis
A high-priority endpoint for lab partners to perform risk assessments on raw numerical data.

- **Endpoint:** `POST /b2b/analyze_bulk`
- **Method:** `POST`
- **Headers:** `x-api-key: YOUR_KEY_HERE`

### Request Body (JSON)
```json
{
  "data": {
    "Glucose": 110,
    "Hemoglobin": 14.5,
    "Cholesterol": 190
  }
}
```

### Success Response
```json
{
  "status": "Success",
  "billed_to": "Diagnostic Lab Partner",
  "critical_alerts": [],
  "analysis": {
    "Diabetes": { "prediction": "Healthy", "risk_score": 15.2 }
  }
}
```