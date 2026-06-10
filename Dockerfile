# Stage 1: Builder — install heavy deps once, then copy to slim runtime
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies for Pillow and PyTorch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime — copy only what we need
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime OS libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and assets
COPY src/         ./src/
COPY configs/     ./configs/
COPY models/      ./models/

# Non-root user for security best practice
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Environment variable for config path
ENV CONFIG_PATH=configs/config.yaml

# Health check — Docker will mark container unhealthy if /health returns non-200
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start Uvicorn ASGI server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]