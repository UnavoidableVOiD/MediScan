import os
import glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. SETUP PATHS
script_dir = os.path.dirname(os.path.abspath(__file__))
kb_path = os.path.join(script_dir, "../../knowledge_base")
db_path = os.path.join(script_dir, "../../vector_store")

def create_vector_db():
    print(f"Scanning for medical documents in: {kb_path}")
    
    documents = []
    
    #load txt files
    txt_files = glob.glob(os.path.join(kb_path, "*.txt"))
    for file in txt_files:
        print(f"Loading Text: {os.path.basename(file)}")
        loader = TextLoader(file)
        documents.extend(loader.load())

    #load pdf files
    pdf_files = glob.glob(os.path.join(kb_path, "*.pdf"))
    for file in pdf_files:
        print(f"   - Loading PDF: {os.path.basename(file)}")
        loader = PyPDFLoader(file)
        documents.extend(loader.load())

    if not documents:
        print("No documents found! Add .txt or .pdf files to 'knowledge_base/'")
        return

    print(f"Total Pages/Documents Loaded: {len(documents)}")

    #splitting texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} searchable chunks.")

    #embeddings
    print("Generatings Embeddings (This may take time for large PDFs)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(db_path)
    print(f"Knowledge Base successfully saved to: {db_path}")

if __name__ == "__main__":
    create_vector_db()