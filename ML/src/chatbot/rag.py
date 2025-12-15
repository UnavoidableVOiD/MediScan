import os
import sys
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        
        #HYBRID ENGINE SELECTOR
        groq_api_key = os.getenv("GROQ_API_KEY")
        use_cloud = os.getenv("USE_CLOUD_LLM", "False").lower() == "true"

        self.llm = None

        if use_cloud and groq_api_key and ChatGroq:
            print("\n⚡ MODE: CLOUD (Groq Llama-3)")
            try:
                self.llm = ChatGroq(
                    temperature=0.2,
                    model_name="llama-3.1-8b-instant",
                    api_key=groq_api_key
                )
                print("✓ Connected to Groq Cloud.")
            except Exception as e:
                print(f"!! Groq Error: {e}")

        if not self.llm:
            print("\nMODE: LOCAL (Ollama Llama-3.2)")
            try:
                self.llm = ChatOllama(
                    model="llama3.2",    
                    temperature=0.2,      
                    base_url="http://localhost:11434"
                )
                print("✓ Connected to Local Ollama.")
            except Exception as e:
                print(f"!! Ollama Connection Failed: {e}")
                print("Make sure you ran 'ollama run llama3.2' in terminal!")

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

    def ask(self, query, patient_data=None):
        if not self.llm:
            return "System Error: No AI Engine is running (Check Ollama or Groq Key)."
        
        if not self.db:
            return "System Error: Knowledge Base unavailable."

        try:
            docs = self.retriever.invoke(query)
            context_text = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            context_text = "No specific medical guidelines found."

        patient_context_str = "No specific patient report uploaded."
        if patient_data and isinstance(patient_data, dict):
            patient_context_str = "CURRENT PATIENT REPORT VALUES:\n"
            for key, val in patient_data.items():
                patient_context_str += f"- {key}: {val}\n"

        prompt_template = ChatPromptTemplate.from_template("""
        You are MediScan-Bot, an expert medical assistant.
        
        <MEDICAL_GUIDELINES>
        {context}
        </MEDICAL_GUIDELINES>
        
        <PATIENT_REPORT_DATA>
        {patient_info}
        </PATIENT_REPORT_DATA>

        USER QUESTION: {question}

        INSTRUCTIONS:
        1. Base your answer strictly on the MEDICAL GUIDELINES and PATIENT REPORT DATA.
        2. If the patient's data shows abnormal values, explicitily mention them.
        3. Be concise and professional.
        4. DISCLAIMER: Never give a definitive diagnosis. Always say "This suggests..." or "Consult a doctor."

        ANSWER:
        """)

        chain = prompt_template | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "context": context_text,
            "patient_info": patient_context_str,
            "question": query
        })
        
        return response

if __name__ == "__main__":
    bot = MedicalChatbot()
    print(bot.ask("What is a normal glucose level?"))