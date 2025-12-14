from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.inference import DiseasePredictor
from models.safety import SafetyGuard
from ocr_engine.ocr_main import extract_text_with_layout, parse_lab_report
from generation.report_generator import MedicalReportGenerator
from chatbot.rag import MedicalChatbot

app = FastAPI(title="MediScan AI Core (Revenue Enabled)")

print("Initializing AI Services...")
engine = DiseasePredictor()
report_gen = MedicalReportGenerator()
safety_guard = SafetyGuard()
chatbot = MedicalChatbot()
print("System Ready.")

class ManualDataRequest(BaseModel):
    data: Dict[str, Any]

class ChatRequest(BaseModel):
    question: str
    patient_context: Optional[Dict[str, Any]] = None
    #freemium check
    is_premium: bool = False 

#auth helpers
async def verify_api_key(x_api_key: str = Header(None)):
    """
    Simulates checking a B2B Partner's paid API Key.
    """
    valid_keys = ["LAB_PARTNER_123", "HOSPITAL_X_456"]
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid or Missing B2B API Key")
    return x_api_key

@app.get("/")
def home():
    return {"status": "MediScan AI is Online", "revenue_mode": "Active"}

#free tier(patient upload)
@app.post("/analyze_report")
async def analyze_pdf_report(file: UploadFile = File(...)):
    """
    Standard analysis. Free for patients.
    """
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. OCR Extraction
        lines = extract_text_with_layout(temp_filename)
        extracted_data = parse_lab_report(lines)
        

        #if OCR fails to read data,(STOP)
        if not extracted_data:
            if os.path.exists(temp_filename): os.remove(temp_filename)
            return {
                "status": "Failed",
                "error": "OCR could not extract any data. Please ensure the PDF contains readable text (Glucose, Creatinine, etc).",
                "debug_ocr_preview": lines[:10] if lines else "No text found."
            }

        #pipeline
        critical_alerts = safety_guard.check_criticals(extracted_data)
        health_analysis = engine.analyze_full_report(extracted_data)
        
        #report generation
        #pass 'critical_alerts' so the AI knows to panic
        patient_text = report_gen.generate_patient_summary(
            health_analysis, 
            raw_data=extracted_data, 
            critical_alerts=critical_alerts
        )
        
        doctor_text = report_gen.generate_doctor_summary(
            health_analysis, 
            raw_data=extracted_data, 
            critical_alerts=critical_alerts
        )
        
        if critical_alerts:
            patient_text = "\n**URGENT:** Critical values detected!\n" + patient_text
        
        os.remove(temp_filename)
        
        return {
            "source": "ocr_extraction",
            "tier": "Free User",
            "raw_data": extracted_data,
            "critical_alerts": critical_alerts,
            "health_analysis": health_analysis,
            "summary_for_patient": patient_text,
            "summary_for_doctor": doctor_text
        }
    except Exception as e:
        if os.path.exists(temp_filename): os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))

#premium tier(chatbot)
@app.post("/chat")
def chat_with_medibot(request: ChatRequest):
    """
    [REVENUE GATE] Only allows access if is_premium=True.
    """
    if not request.is_premium:
        return {
            "error": "Premium Feature Locked",
            "message": "To chat with MediBot AI, please upgrade to MediScan Premium.",
            "upgrade_url": "/subscribe"
        }

    #if Premium, allow access to RAG
    response = chatbot.ask(request.question, patient_data=request.patient_context)
    
    return {
        "question": request.question, 
        "answer": response,
        "tier": "Premium User"
    }

#b2b tier(paid API for labs)
@app.post("/b2b/analyze_bulk", dependencies=[Depends(verify_api_key)])
def b2b_analysis(request: ManualDataRequest):
    """
    [REVENUE GATE] Paid endpoint for Diagnostic Labs.
    Requires 'x-api-key' header.
    """
    data = request.data
    
    critical_alerts = safety_guard.check_criticals(data)
    health_analysis = engine.analyze_full_report(data)
    
    return {
        "status": "Success",
        "billed_to": "Diagnostic Lab Partner",
        "critical_alerts": critical_alerts,
        "analysis": health_analysis
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)