# Lightweight Python base for the Nanodynamics-v3 Flight-Director console
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first for fast build and small image footprint (~150MB instead of ~2GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose Render default port
EXPOSE 8000

# Start the v3 FastAPI / Uvicorn web console respecting Render's dynamic $PORT environment variable
CMD uvicorn web_console.server:app --host 0.0.0.0 --port ${PORT:-8000}
