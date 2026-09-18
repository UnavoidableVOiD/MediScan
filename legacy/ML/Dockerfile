
FROM python:3.11-slim as builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy Requirements
COPY requirements.txt .

# 1. Install CPU Torch specifically
# We use --prefix to install into a specific folder we can copy later
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install everything else
# We add the extra-index-url here too, just in case a dependency tries to update torch
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

    #final image production
FROM python:3.11-slim

WORKDIR /app

# 1. Install Runtime System Dependencies (Tesseract + GL)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the installed Python packages from the Builder Stage
COPY --from=builder /install /usr/local

# 3. Copy Source Code
COPY . .

# 4. Run
EXPOSE 8001
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]