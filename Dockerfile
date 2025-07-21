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
    git-lfs \
    libasound2-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ Initialize Git LFS
RUN git lfs install

# --- Install Python dependencies ---
COPY requirements-core.txt .
COPY requirements-heavy.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir -r requirements-heavy.txt

# ✅ Copy all backend source files
COPY . .

# === Frontend build stage ===
FROM node:18 AS frontend-build

WORKDIR /frontend

# --- Install frontend dependencies ---
COPY frontend_part/project/package*.json ./
RUN npm install

# --- Copy and build frontend ---
COPY frontend_part/project ./
RUN npm run build

# === Final stage for deployment ===
FROM python:3.10-slim

WORKDIR /app

# --- Reinstall runtime system dependencies ---
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
    git-lfs \
    libasound2-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# --- Install Python dependencies ---
COPY requirements-core.txt .
COPY requirements-heavy.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir -r requirements-heavy.txt

# ✅ Copy everything from base stage
COPY --from=base /app /app

# ✅ Copy built frontend
COPY --from=frontend-build /frontend/dist /app/frontend_dist

# ✅ Make sure these folders exist (avoid FileNotFoundError)
RUN mkdir -p /app/audio /app/video /app/models/buffalo_l /app/utils

# ✅ Permissions for audio/video if needed
RUN chmod -R 755 /app/audio /app/video /app/models /app/utils

# --- Expose Streamlit port ---
EXPOSE 8501

# ✅ Start Streamlit headlessly
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
