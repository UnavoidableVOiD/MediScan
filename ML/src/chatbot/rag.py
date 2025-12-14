import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MedicalChatbot:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.script_dir, "../../vector_store")
        
        #INITIALIZE LOCAL LLM (Llama 3.2 via Ollama)
        print("Initializing Local Llama-3.2...")
        try:
            self.llm = ChatOllama(
                model="llama3.2",    
                temperature=0.2,      
                base_url="http://localhost:11434"
            )
            print("✓ Local Inference Engine Connected.")
        except Exception as e:
            print(f"!! Ollama Connection Failed: {e}")
            print("Make sure you ran 'ollama run llama3.2' in terminal!")
            self.llm = None

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
            return "System Error: Local AI Engine (Ollama) is not running."
        
        if not self.db:
            return "System Error: Knowledge Base unavailable."

        #Finds top 3 relevant paragraphs from WHO/KDIGO PDFs
        docs = self.retriever.get_relevant_documents(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        patient_context_str = "No specific patient report uploaded."
        if patient_data and isinstance(patient_data, dict):
            # Convert JSON {"Glucose": 160}
            patient_context_str = "CURRENT PATIENT REPORT VALUES:\n"
            for key, val in patient_data.items():
                patient_context_str += f"- {key}: {val}\n"

        #construct prompt
        prompt_template = ChatPromptTemplate.from_template("""
        You are MediScan-Bot, an expert medical assistant.
        
        <MEDICAL_GUIDELINES>
        (Verified facts from WHO/KDIGO)
        {context}
        </MEDICAL_GUIDELINES>
        
        <PATIENT_REPORT_DATA>
        (Specific values from the user's lab test)
        {patient_info}
        </PATIENT_REPORT_DATA>

        USER QUESTION: {question}

        INSTRUCTIONS:
        1. Base your answer strictly on the MEDICAL GUIDELINES and PATIENT REPORT DATA.
        2. If the patient's data shows abnormal values, explicitily mention them (e.g., "Your Glucose is 160...").
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
    dummy_report = {"Glucose": 160, "Age": 45}
    print("\n--- TEST SESSION (Local Llama 3.2) ---")
    
    while True:
        q = input("\nAsk MediBot (or 'q'): ")
        if q.lower() == 'q': break
        print(bot.ask(q, patient_data=dummy_report))