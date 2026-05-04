FROM python:3.11-slim

# Install Java (required for PySpark)
RUN apt-get update && apt-get install -y \
    default-jdk-headless \
    procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYSPARK_PYTHON=python3

WORKDIR /app

COPY requirements.txt .

# Install numpy first, then the rest
RUN pip install --no-cache-dir numpy==1.26.4 && \
    pip install --no-cache-dir flask==3.0.3 gunicorn==21.2.0 requests==2.31.0 && \
    pip install --no-cache-dir pyspark==3.5.3 && \
    pip install --no-cache-dir python-telegram-bot==20.7

COPY app.py .
COPY bot.py .
COPY models/ ./models/

EXPOSE 5000

CMD ["gunicorn", "--workers=1", "--threads=4", "--timeout=120", "--bind=0.0.0.0:5000", "app:app"]