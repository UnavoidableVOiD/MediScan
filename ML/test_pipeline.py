import requests
import json
import os

# CONFIG
API_URL = "http://localhost:8000"
TEST_PDF = "datasets/test_reports/Report2.pdf" 

def test_analyze():
    print(f"\nTEST 1: Uploading {TEST_PDF} to AI Engine...")
    
    if not os.path.exists(TEST_PDF):
        print(f"Error: File {TEST_PDF} not found. Check path!")
        return None

    with open(TEST_PDF, 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post(f"{API_URL}/analyze_report", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("\nANALYSIS SUCCESS!")
                print("------------------------------------------------")
                print(f"Health Predictions: {json.dumps(data['health_analysis'], indent=2)}")
                print("------------------------------------------------")
                print(f"Critical Alerts: {data['critical_alerts']}")
                return data['raw_data']
            else:
                print(f"Server Error: {response.text}")
                return None
        except Exception as e:
            print(f"Connection Failed: {e}")
            return None

def test_chat(extracted_data):
    print("\n\nTEST 2: Testing RAG Chatbot with Context...")
    
    if not extracted_data:
        print("Skipping Chat test (No data from Step 1)")
        return

    payload = {
        "question": "What does my Creatinine level indicate?",
        "patient_context": extracted_data,
        "is_premium": True
    }
    
    try:
        response = requests.post(f"{API_URL}/chat", json=payload)
        res_json = response.json()
        print(f"\nMediBot Says:\n{res_json['answer']}")
    except Exception as e:
        print(f"Chat Failed: {e}")

if __name__ == "__main__":
    # 1. Run Analysis
    ocr_data = test_analyze()
    
    # 2. If Analysis worked, ask the Chatbot about it
    if ocr_data:
        test_chat(ocr_data)