# Dockerfile for Python Voice Service
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies for audio (optional, but good practice)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY voice_service.py .

# Expose the port the WebSocket server will listen on
EXPOSE 8765

# Command to run the application
# Ensure environment variables are set when running the container
CMD ["python", "voice_service.py"]
