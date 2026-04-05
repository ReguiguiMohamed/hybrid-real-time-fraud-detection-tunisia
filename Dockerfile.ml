FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (Java for PySpark)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    openjdk-21-jdk \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sfn /usr/share/zoneinfo/UTC /etc/localtime

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "src/ml/train_model.py"]
