import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_models import ChatOllama
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

class MedicalReportGenerator:
    def __init__(self):
        load_dotenv()
        self.llm = None
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        use_cloud = os.getenv("USE_CLOUD_LLM", "False").lower() == "true"

        if use_cloud and groq_api_key and ChatGroq:
            print("Report Gen: Using Groq Cloud (Llama-3)...")
            try:
                self.llm = ChatGroq(
                    temperature=0.3,
                    model_name="llama-3.1-8b-instant", 
                    api_key=groq_api_key
                )
            except Exception as e:
                print(f"!! Groq Error: {e}")

        #Local (Ollama)
        if not self.llm:
            print("Report Gen: Using Local Ollama (Llama-3.2)...")
            try:
                self.llm = ChatOllama(
                    model="llama3.2",
                    temperature=0.3, 
                    base_url="http://localhost:11434"
                )
            except Exception as e:
                print(f"!! Ollama Connection Failed: {e}")
                self.llm = None

    def _generate_llm_summary(self, analysis, raw_data, critical_alerts, audience="patient"):
        if not self.llm: 
            return "Error: AI Inference Engine is unavailable."

        #format Analysis for AI Reading
        analysis_str = ""
        for disease, result in analysis.items():
            status = result.get('status', result.get('prediction', 'Unknown'))
            risk = result.get('risk_score', 0)
            analysis_str += f"- {disease}: {status} (Risk Probability: {risk}%)\n"
        
        #Critical Alerts (Prioritization)
        alert_str = "NONE"
        if critical_alerts:
            alert_str = "\n".join([f"CRITICAL: {a['parameter']} is {a['value']} ({a['message']})" for a in critical_alerts])

        if audience == "patient":
            system_prompt = """
            You are MediScan, a compassionate medical AI assistant.
            
            <SAFETY_PROTOCOL>
            The following Critical Alerts were detected by the Safety Guard:
            {alerts}
            
            IF alerts exist:
            1. Start IMMEDIATELY with: "URGENT ATTENTION NEEDED"
            2. Explain the critical value clearly and calmly.
            3. Tell the user to consult a doctor immediately.
            4. IGNORE any "Healthy" predictions for the affected organ.
            </SAFETY_PROTOCOL>

            <REPORT_CONTEXT>
            AI Disease Predictions:
            {analysis}

            Raw Lab Values:
            {raw_data}
            </REPORT_CONTEXT>

            INSTRUCTIONS:
            - Write a summary for a PATIENT (Non-medical background).
            - Use simple, reassuring language.
            - Explain what the abnormal values mean.
            - Keep it under 150 words.
            """
        else: # Doctor
            system_prompt = """
            You are a Clinical Pathologist AI assistant.
            
            <CRITICAL_FLAGS>
            {alerts}
            </CRITICAL_FLAGS>

            <DIAGNOSTIC_DATA>
            ML Predictions:
            {analysis}

            Raw Lab Values:
            {raw_data}
            </DIAGNOSTIC_DATA>

            INSTRUCTIONS:
            - Write a clinical summary for a GENERAL PHYSICIAN.
            - Start with "Clinical Impression:".
            - Use medical terminology.
            - Highlight correlations between abnormal values and disease risks.
            - Format with bullet points.
            """

        try:
            prompt = ChatPromptTemplate.from_template(system_prompt)
            chain = prompt | self.llm | StrOutputParser()

            return chain.invoke({
                "analysis": analysis_str, 
                "raw_data": str(raw_data), 
                "alerts": alert_str
            })
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def generate_patient_summary(self, health_analysis, raw_data=None, critical_alerts=None):
        return self._generate_llm_summary(health_analysis, raw_data, critical_alerts, audience="patient")

    def generate_doctor_summary(self, health_analysis, raw_data=None, critical_alerts=None):
        return self._generate_llm_summary(health_analysis, raw_data, critical_alerts, audience="doctor")