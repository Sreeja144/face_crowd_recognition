# === Base image for Python backend ===
FROM python:3.10-slim AS base

WORKDIR /app

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libasound2-dev \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt .
COPY requirements-heavy.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir -r requirements-heavy.txt

COPY . .

# ✅ Download ONNX model instead of using Git LFS
RUN mkdir -p models/buffalo_l && \
    wget -O models/buffalo_l/1k3d68.onnx https://huggingface.co/nttc-ai/insightface-models/resolve/main/models/buffalo_l/1k3d68.onnx

# === Frontend build stage ===
FROM node:18 AS frontend-build

WORKDIR /frontend

COPY frontend_part/project/package*.json ./
RUN npm install

COPY frontend_part/project ./
RUN npm run build

# === Final stage for deployment ===
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libasound2-dev \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt .
COPY requirements-heavy.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir -r requirements-heavy.txt

COPY --from=base /app /app
COPY --from=frontend-build /frontend/dist /app/frontend_dist

COPY .streamlit /app/.streamlit

RUN mkdir -p /app/audio /app/video /app/models/buffalo_l /app/utils
RUN chmod -R 755 /app/audio /app/video /app/models /app/utils
# Make the script executable
RUN chmod +x download_models.sh

# Run the script before starting the app
RUN ./download_models.sh

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]

