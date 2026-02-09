# MediScan

**MediScan** is an AI-powered diagnostic platform. This repository contains the **Intelligence Engine** (Backend Microservice) responsible for:

1.  **Medical OCR:** Extracting structured data from raw Lab Report PDFs (scanned/digital).
2.  **Disease Prediction:** Running 6 "Recall-Optimized" ML models (Diabetes, Heart, Kidney, Liver, Thyroid, Anemia).
3.  **Safety Guardrails:** Auto-correcting OCR typos and flagging life-threatening values.
4.  **RAG Chatbot:** A context-aware medical assistant using **Llama-3** (Local or Cloud).

---

## Quick Start (Docker - Recommended)

The easiest way to run the AI Core is via Docker. This ensures all system dependencies (Tesseract, OpenCV, Poppler) are pre-installed.

### 1. Prerequisites
*   **Docker Desktop** installed.
*   **Ollama** installed on your host machine (for the Chatbot).
    *   Run `ollama pull llama3.2` to download the model.

### 2. Prepare Ollama (Crucial Step)
By default, Ollama blocks Docker connections. You must run it in "Public" mode on your Mac/PC:

1.  **Quit Ollama** from the top menu bar (System Tray).
2.  Run this in a terminal window (and keep it open):
    ```bash
    OLLAMA_HOST=0.0.0.0 ollama serve
    ```

### 3. Build & Run
Open a new terminal in this project folder:

```bash
# 1. Build the Image
docker build -t mediscan-ml .

# 2. Run the Container
# Note: host.docker.internal allows Docker to talk to your Mac's Ollama
docker run -p 8001:8001 \
  --name mediscan_container \
  --env OLLAMA_BASE_URL="http://host.docker.internal:11434" \
  --env USE_CLOUD_LLM="False" \
  mediscan-ml