import os
import sys
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections.abc import AsyncGenerator

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

class MedicalChatbot:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.script_dir, "../../vector_store")
        
        # HYBRID ENGINE SELECTOR
        groq_api_key = os.getenv("GROQ_API_KEY")
        use_cloud = os.getenv("USE_CLOUD_LLM", "False").lower() == "true"

        self.llm = None

        if use_cloud and groq_api_key and ChatGroq:
            print("\n⚡ MODE: CLOUD (Groq Llama-3)")
            try:
                self.llm = ChatGroq(
                    temperature=0.1, 
                    model_name="llama-3.1-8b-instant",
                    api_key=groq_api_key
                )
                print("✓ Connected to Groq Cloud.")
            except Exception as e:
                print(f"!! Groq Error: {e}")

        if not self.llm:
            print("\nMODE: LOCAL (Ollama Llama-3.2)")
            try:
                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self.llm = ChatOllama(
                    model="llama3.2",    
                    temperature=0.0,     
                    base_url=ollama_url
                )
                print("✓ Connected to Local Ollama.")
            except Exception as e:
                print(f"!! Ollama Connection Failed: {e}")

        print("Loading Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} 
        )
        
        try:
            self.db = FAISS.load_local(
                self.db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            self.retriever = self.db.as_retriever(search_kwargs={"k": 3})
            print("✓ MediBot Knowledge Base Loaded.")
        except Exception as e:
            print(f"!! Error loading Vector DB: {e}")
            self.db = None

    def _prepare_context(self, query, patient_data):
        try:
            docs = self.retriever.invoke(query)
            context_text = "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            context_text = "No specific medical guidelines found."

        patient_context_str = "No specific patient report uploaded."
        
        if patient_data and isinstance(patient_data, dict):
            patient_context_str = "## CURRENT PATIENT REPORT SUMMARY ##\n"
            
            lab_values = patient_data.get('lab_values', {})
            if lab_values:
                patient_context_str += "\n### LABORATORY VALUES:\n"
                for key, val in lab_values.items():
                    patient_context_str += f"- {key}: {val}\n"
            
            analysis = patient_data.get('analysis', {})
            if analysis:
                patient_context_str += "\n### AI ANALYSIS FINDINGS:\n"
                patient_context_str += f"- RISK LEVEL: {analysis.get('risk_level', 'N/A')}\n"
                
                conditions = analysis.get('conditions', [])
                if conditions:
                    safe_conditions = []
                    for c in conditions:
                        if isinstance(c, dict):
                            val = c.get('condition') or c.get('prediction') or c.get('name') or str(c)
                            safe_conditions.append(str(val))
                        else:
                            safe_conditions.append(str(c))
                    patient_context_str += f"- DETECTED CONDITIONS: {', '.join(safe_conditions)}\n"
                
                summary = analysis.get('summary', '')
                if summary:
                    patient_context_str += f"- CLINICAL SUMMARY: {summary}\n"
        
        return context_text, patient_context_str

    def _get_chain(self):
        """Returns the LangChain pipeline"""
        prompt_template = ChatPromptTemplate.from_template("""
        SYSTEM: You are MediScan, a professional AI Medical Assistant.
        
        CONTEXT:
        {context}
        
        PATIENT REPORT:
        {patient_info}

        --- INSTRUCTIONS ---
        1. IF User says "Hi", "Hello": Reply EXACTLY: "Hello! I am MediScan. I have analyzed your report. How can I help you?". DO NOT USE THE INTRODUCTION AFTERWARDS AS YOU ALREADY HAVE GREETED THE USER. 
        2. IF User asks "How is my report?" or medical questions: 
           - Answer DIRECTLY. 
           - DO NOT say "Here is the summary" or use '###' headers. 
           - Simply state the findings (e.g. "Your Liver values are elevated...").
        3. IF User asks non-medical (movies, recipes): Reply EXACTLY: "I apologize, but I am specialized only in Medical Analysis."
        4. ALWAYS end with "Consult a doctor."

        --- EXAMPLES ---
        User: "Hi"
        MediScan: "Hello! I am MediScan. I have analyzed your report. How can I help you?"

        User: "How is my report?"
        MediScan: "Your report indicates potential Liver issues due to elevated Alkaline Phosphatase. Your Heart risk is also elevated. Please consult a doctor."

        User: "Who is Batman?"
        MediScan: "I apologize, but I am specialized only in Medical Analysis."

        ----------------

        User: {question}
        MediScan:
        """)
        return prompt_template | self.llm | StrOutputParser()

    def ask(self, query, patient_data=None):
        if not self.llm or not self.db: return "System Error: AI unavailable."
        context, patient_info = self._prepare_context(query, patient_data)
        chain = self._get_chain()
        return chain.invoke({"context": context, "patient_info": patient_info, "question": query})

    async def stream_ask(self, query, patient_data=None) -> AsyncGenerator[str, None]:
        if not self.llm:
            yield "System Error: AI Engine unavailable."
            return
        if not self.db:
            yield "System Error: Knowledge Base unavailable."
            return

        context, patient_info = self._prepare_context(query, patient_data)
        chain = self._get_chain()
        async for chunk in chain.astream({"context": context, "patient_info": patient_info, "question": query}):
            yield chunk

if __name__ == "__main__":
    bot = MedicalChatbot()
    print("Testing Sync Ask:")
    print(bot.ask("What is a normal glucose level?"))