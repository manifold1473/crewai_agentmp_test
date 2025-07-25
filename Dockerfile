# Use official Python image as base
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies (if needed for crewai_tools, e.g., curl, build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy .env if present (optional, for local dev)
COPY .env .env

# Expose the port the app runs on
EXPOSE 8080

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Default command to run the agent server
CMD ["python", "-m", "main"] 