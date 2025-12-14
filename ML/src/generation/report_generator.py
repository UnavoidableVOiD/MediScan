import os
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MedicalReportGenerator:
    def __init__(self):
        print("Initializing Generative Report Engine (Llama 3.2)...")
        try:
            self.llm = ChatOllama(
                model="llama3.2",
                temperature=0.3, 
                base_url="http://localhost:11434"
            )
        except Exception:
            self.llm = None

    def _generate_llm_summary(self, analysis, raw_data, critical_alerts, audience="patient"):
        if not self.llm: return "AI unavailable."

        analysis_str = ""
        for disease, result in analysis.items():
            analysis_str += f"- {disease}: {result.get('prediction')} (Risk Score: {result.get('risk_score')}%, Status: {result.get('status', 'N/A')})\n"
        
        # Format Critical Alerts for the AI
        alert_str = "NONE"
        if critical_alerts:
            alert_str = "\n".join([f"!!! CRITICAL ALERT: {a['parameter']} is {a['value']} ({a['message']})" for a in critical_alerts])

        if audience == "patient":
            system_prompt = """
            You are MediScan, a medical assistant. 
            
            <CRITICAL_WARNING>
            The following values are DANGEROUSLY HIGH/LOW. You MUST start your summary by mentioning these immediately.
            Alerts: {alerts}
            If alerts exist, IGNORE any "Healthy" prediction from the ML model for that organ. The raw data implies immediate danger.
            </CRITICAL_WARNING>

            INPUT DATA:
            1. ML Predictions: {analysis}
            2. Raw Lab Values: {raw_data}

            INSTRUCTIONS:
            - If Critical Alerts exist: Start with "URGENT: Your report shows critical values for [Parameter]."
            - Then, summarize the rest of the report.
            - Explain what the values mean in simple English.
            - End with: "Please visit a hospital immediately."
            - Keep it under 150 words.
            """
        else: # Doctor
            system_prompt = """
            You are a Clinical Pathologist AI. Summarize results for a GP.

            <CRITICAL_WARNING>
            IMMEDIATE ACTION REQUIRED FOR:
            {alerts}
            </CRITICAL_WARNING>

            INPUT DATA:
            1. ML Predictions: {analysis}
            2. Raw Lab Values: {raw_data}

            INSTRUCTIONS:
            - Start with "CRITICAL FINDINGS" if alerts exist.
            - Provide a Clinical Impression.
            - Correlate Raw Values with Pathologies (e.g., "Creatinine 9.0 -> End Stage Renal Failure").
            - Be professional and concise.
            """

        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "analysis": analysis_str, 
            "raw_data": str(raw_data), 
            "alerts": alert_str
        })

    def generate_patient_summary(self, health_analysis, raw_data=None, critical_alerts=None):
        return self._generate_llm_summary(health_analysis, raw_data, critical_alerts, audience="patient")

    def generate_doctor_summary(self, health_analysis, raw_data=None, critical_alerts=None):
        return self._generate_llm_summary(health_analysis, raw_data, critical_alerts, audience="doctor")