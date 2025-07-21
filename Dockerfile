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

RUN git lfs install

COPY requirements-core.txt .
COPY requirements-heavy.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir -r requirements-heavy.txt

COPY . .

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
    git-lfs \
    libasound2-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git lfs install

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

CMD sh -c "streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"

