# ==============================================================================
# Nanodynamics Platform & OrbitGuard AI Production Dockerfile
# Multi-stage build optimized for Render Cloud & Containerized Deployments
# ==============================================================================

# Stage 1: Build stage for dependency compilation
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /install

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only wheel to keep image lightweight (<250MB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install/env torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install python dependencies into prefix using cpu extra-index
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/env -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu


# ==============================================================================
# Stage 2: Production runtime image
# ==============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    ENVIRONMENT=production

WORKDIR /app

# Install runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create secure non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 -d /app appuser

# Copy installed Python packages from builder
COPY --from=builder /install/env /usr/local

# Copy application source code
COPY . /app

# Ensure proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Render / Container default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Launch production Uvicorn server
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
