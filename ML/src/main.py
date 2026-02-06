from fastapi import FastAPI, UploadFile, File, HTTPException
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

class AnalysisRequest(BaseModel):
    """
    Step 2 Input: The user sends back the (potentially corrected) data.
    """
    patient_data: Dict[str, Any]
    user_id: Optional[str] = "guest"

class ChatRequest(BaseModel):
    question: str
    patient_context: Optional[Dict[str, Any]] = None
    #freemium check
    is_premium: bool = False 


@app.get("/")
def home():
    return {"status": "MediScan AI is Online", "mode": "Human-Verification Enabled"}

#OCR ONLY 
@app.post("/extract_from_pdf")
async def extract_data_step1(file: UploadFile = File(...)):
    """
    Step 1: Upload PDF -> Return Raw JSON.
    The Frontend should display this JSON in a form for the user to edit/verify.
    """
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        #run vision pipeline
        lines = extract_text_with_layout(temp_filename)
        raw_extracted_data = parse_lab_report(lines)
        
        #cleanup
        if os.path.exists(temp_filename): os.remove(temp_filename)

        if not raw_extracted_data:
            return {
                "status": "Failed",
                "message": "Could not read data. Please enter values manually.",
                "ocr_debug": lines[:5] if lines else []
            }
        
        #sanitization for correctly checking data
        #eg fix "44" -> "4.4" so the user doesn't have to.
        auto_corrected_data = safety_guard.sanitize_data(raw_extracted_data)

        return {
            "status": "Success",
            "message": "Please review these values before analysis.",
            "extracted_data": auto_corrected_data
        }

    except Exception as e:
        if os.path.exists(temp_filename): os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_verified_data")
def analyze_data_step2(request: AnalysisRequest):
    """
    Step 2: User submits Verified Data -> AI runs Diagnosis.
    """
    data = request.patient_data
    
    #panic values
    critical_alerts = safety_guard.check_criticals(data)
    
    #inference
    health_analysis = engine.analyze_full_report(data)
    
    #llm report generation
    patient_text = report_gen.generate_patient_summary(
        health_analysis, 
        raw_data=data, 
        critical_alerts=critical_alerts
    )
    
    doctor_text = report_gen.generate_doctor_summary(
        health_analysis, 
        raw_data=data, 
        critical_alerts=critical_alerts
    )
    
    if critical_alerts:
        patient_text = "URGENT: CRITICAL VALUES DETECTED!\n\n" + patient_text

    return {
        "status": "Analysis Complete",
        "critical_alerts": critical_alerts,
        "risk_assessment": health_analysis,
        "summary_patient": patient_text,
        "summary_doctor": doctor_text
    }


@app.post("/chat")
def chat_with_medibot(request: ChatRequest):
    if not request.is_premium:
        return {"error": "Premium Feature Locked", "message": "Upgrade to chat."}

    response = chatbot.ask(request.question, patient_data=request.patient_context)
    return {"answer": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)