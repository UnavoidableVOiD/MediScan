# 📘 MediScan AI API Documentation

**Base URL:** `http://localhost:8000`

---

## 1. Analyze Lab Report (OCR + AI Prediction)
Uploads a PDF, extracts text, checks for critical values, and runs ML models.

- **Endpoint:** `POST /analyze_report`
- **Headers:** `Content-Type: multipart/form-data`
- **Body:**
  - `file`: (File Object) The PDF report.

### Success Response (JSON)
```json
{
  "tier": "Free User",
  "critical_alerts": [
    "CRITICAL: Potassium is 7.0 (Immediate Cardiac Risk)" 
    // Array is empty if patient is safe
  ],
  "health_analysis": {
    "Diabetes": { "prediction": "Diabetic", "risk_score": 88.5 },
    "Kidney": { "prediction": "Healthy", "risk_score": 12.0 }
  },
  "summary_for_patient": "URGENT: Your potassium is critically high...",
  "summary_for_doctor": "Clinical Impression: Hyperkalemia detected...",
  "raw_data": {
    "Glucose": 160,
    "Creatinine": 0.9
    // FRONTEND MUST SAVE THIS RAW_DATA FOR CHATBOT CONTEXT
  }
}