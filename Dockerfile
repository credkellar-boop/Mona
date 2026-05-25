# Dockerfile
FROM python:3.10-slim

# Install system dependencies for OpenCV and WebRTC
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository
COPY . .

# Expose the WebRTC SFU port
EXPOSE 8080

# Default command to spin up the live signaling server
CMD ["python", "live_server.py"]
