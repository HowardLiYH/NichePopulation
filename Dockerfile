# Dockerfile for reproducible experiments
# NeurIPS 2025: Emergent Specialization in Multi-Agent Trading

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install package
RUN pip install -e .

# Create results directories
RUN mkdir -p results paper/figures paper/tables

# Default command: run quick test
CMD ["python", "-m", "experiments.runner", "-e", "exp1", "--trials", "10"]

# To run full experiments:
# docker run emergent-specialization python -m experiments.runner --all --trials 100

# To run specific experiment:
# docker run emergent-specialization python experiments/exp1_emergence.py --trials 100
