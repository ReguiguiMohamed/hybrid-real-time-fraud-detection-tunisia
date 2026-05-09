FROM python:3.11-slim

# Create non-root user (HF requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# System dependencies
USER root
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
USER user

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code with correct ownership
COPY --chown=user . .

# Expose the port HF expects
EXPOSE 7860

# Use port 7860 as required by Hugging Face
CMD ["uvicorn", "dashboard.api:app", "--host", "0.0.0.0", "--port", "7860"]