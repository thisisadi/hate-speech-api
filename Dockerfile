# Use Python 3.11 slim base
FROM python:3.11-slim

# Install Java (required for PySpark)
RUN apt-get update && apt-get install -y \
    default-jdk-headless \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Java home
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY bot.py .

# Copy trained models
COPY models/ ./models/

# Expose port
EXPOSE 5000

# Use gunicorn for production serving
# Single worker because Spark session is not thread-safe across forks
CMD ["gunicorn", "--workers=1", "--threads=4", "--timeout=120", "--bind=0.0.0.0:5000", "app:app"]
