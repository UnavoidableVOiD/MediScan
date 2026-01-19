## MediScan
## download the requirements.txt file and run 
## pip install -r requirements.txt
## pip install langchain-groq python-dotenv

## For RAG SETUP
## TAKE API KEYS FROM GROQ AND MAKE A .env FILE INSIDE THE ML FOLDER
## ONCE INSIDE, PASTE
## # --- AI CONFIG ---
## Set to 'True' to use Cloud (Groq) for work 
## Set to 'False' to use Local (Ollama) for Demo
## USE_CLOUD_LLM=True

## Groq API Key (Get one free at console.groq.com)
## GROQ_API_KEY=gsk_your_key_here_xyz...

## API Endpoints

### 1. Step 1: Extraction (OCR)
*   **URL:** `POST /extract_from_pdf`
*   **Input:** Multipart Form Data (`file`: PDF)
*   **Output:** JSON containing raw extracted values.
    *   *Frontend Task:* Display this JSON in an editable form so the user can fix mistakes (e.g., if OCR reads "Glucose: 900").

### 2. Step 2: Analysis (Brain)
*   **URL:** `POST /analyze_verified_data`
*   **Input:** JSON (The corrected data from Step 1)
    ```json
    {
      "patient_data": {
        "Glucose": 100.0,
        "Creatinine": 1.1,
        ...
      }
    }
    ```
*   **Output:** Risk Predictions + LLM Summaries.

### 3. Chatbot
*   **URL:** `POST /chat`
*   **Input:**
    ```json
    {
      "question": "What does high glucose mean?",
      "is_premium": true,
      "patient_context": {"Glucose": 140}
    }
    ```