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
                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                
                self.llm = ChatOllama(
                    model="llama3.2",    
                    temperature=0.2,      
                    base_url=ollama_url
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

    def _prepare_context(self, query, patient_data):
        """Helper to prepare prompt inputs"""
        try:
            docs = self.retriever.invoke(query)
            context_text = "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            context_text = "No specific medical guidelines found."

        patient_context_str = "No specific patient report uploaded."
        if patient_data and isinstance(patient_data, dict):
            patient_context_str = "CURRENT PATIENT REPORT VALUES:\n"
            for key, val in patient_data.items():
                patient_context_str += f"- {key}: {val}\n"
        
        return context_text, patient_context_str

    def _get_chain(self):
        """Returns the LangChain pipeline"""
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
        If the user says "Hello", "Hi", or "Hey", introduce yourself as MediScan AI.
        2. Base your answer strictly on the MEDICAL GUIDELINES and PATIENT REPORT DATA.
        3. If the patient's data shows abnormal values, explicitily mention them.
        4. Be concise and professional.
        5. DISCLAIMER: Never give a definitive diagnosis. Always say "This suggests..." or "Consult a doctor."

        ANSWER:
        """)
        return prompt_template | self.llm | StrOutputParser()

    def ask(self, query, patient_data=None):
        """Synchronous method (returns full string)"""
        if not self.llm or not self.db:
            return "System Error: AI unavailable."

        context, patient_info = self._prepare_context(query, patient_data)
        chain = self._get_chain()
        
        return chain.invoke({
            "context": context,
            "patient_info": patient_info,
            "question": query
        })

    async def stream_ask(self, query, patient_data=None) -> AsyncGenerator[str, None]:
        """Asynchronous Generator for Streaming Responses"""
        if not self.llm:
            yield "System Error: AI Engine unavailable."
            return
            
        if not self.db:
            yield "System Error: Knowledge Base unavailable."
            return

        # Prepare context (Blocking IO is acceptable here for simplicity)
        context, patient_info = self._prepare_context(query, patient_data)
        chain = self._get_chain()

        # Stream the chunks
        async for chunk in chain.astream({
            "context": context,
            "patient_info": patient_info,
            "question": query
        }):
            yield chunk

if __name__ == "__main__":
    bot = MedicalChatbot()
    print("Testing Sync Ask:")
    print(bot.ask("What is a normal glucose level?"))