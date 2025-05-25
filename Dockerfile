# Multi-stage build for Mycology Research Pipeline
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/app/.local/bin:$PATH \
    FLASK_APP=app.py

# Create non-root user
RUN useradd -m -u 1000 app && \
    mkdir -p /app/uploads /app/results /app/logs && \
    chown -R app:app /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder --chown=app:app /root/.local /home/app/.local

# Copy application code
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Create necessary directories
RUN mkdir -p uploads results logs static/uploads

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"] 