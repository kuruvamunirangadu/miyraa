# Optimized Multi-stage Dockerfile for Miyraa NLP Emotion Engine
# Optimizations:
# - Multi-stage build for minimal image size
# - Layer caching for faster rebuilds
# - Non-root user for security
# - Health checks and readiness probes
# - Production-ready configuration

# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder

WORKDIR /build

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies to custom location for easy copying
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --target=/install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu && \
    pip install --no-cache-dir --target=/install -r requirements.txt

# Stage 2: Runtime - Minimal image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# Copy application code (only necessary files)
COPY src/ ./src/
COPY outputs/ ./outputs/
# Note: data/ and scripts/ can be volume-mounted if needed

# Create non-root user for security
RUN useradd -m -u 1000 miyraa && \
    chown -R miyraa:miyraa /app && \
    mkdir -p /app/logs && \
    chown miyraa:miyraa /app/logs

USER miyraa

# Environment variables (can be overridden)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=info \
    WORKERS=1 \
    PORT=8000

# Expose port for FastAPI
EXPOSE 8000

# Health check using curl (lighter than requests)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the FastAPI server with production settings
CMD ["sh", "-c", "python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS} --log-level ${LOG_LEVEL}"]
