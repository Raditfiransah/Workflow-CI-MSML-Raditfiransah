FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY mlflow_server.py .
COPY config.py .

FROM python:3.11-slim as runtime

# Create non-root user
RUN useradd -m -s /bin/bash mlflow

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only necessary files from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/mlflow_server.py /app/
COPY --from=builder /app/config.py /app/
# Copy model artifacts from the build context (they are present because MLflow run created them)
COPY saved_model /app/saved_model
# Verify the copy
RUN ls -la /app/saved_model

# Change ownership of app directory to mlflow user
RUN chown -R mlflow:mlflow /app

# Switch to non-root user
USER mlflow

# Expose ports
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["python", "mlflow_server.py"]