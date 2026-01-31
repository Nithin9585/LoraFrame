FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including build tools for insightface)
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY models/ ./models/

# Cloud Run uses PORT environment variable (default 8080)
ENV PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Run uvicorn with PORT from environment
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
