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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ Initialize Git LFS (will only work if files are NOT LFS-tracked or properly downloaded)
RUN git lfs install

# --- Install Python dependencies ---
COPY requirements-core.txt .
COPY requirements-heavy.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir -r requirements-heavy.txt

# --- Copy backend code ---
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

# === Final combined stage ===
FROM python:3.10-slim

WORKDIR /app

# --- Install runtime system dependencies ---
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ Install Git LFS again in final stage
RUN git lfs install

# --- Install runtime Python dependencies ---
COPY requirements-core.txt .
COPY requirements-heavy.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir -r requirements-heavy.txt

# --- Copy backend code from base stage ---
COPY --from=base /app /app

# --- Copy frontend build output to app folder ---
COPY --from=frontend-build /frontend/dist /app/frontend_dist



# --- Expose Streamlit port ---
EXPOSE 8501

# --- Start Streamlit ---
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
