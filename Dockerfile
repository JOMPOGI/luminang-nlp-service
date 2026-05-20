FROM python:3.10-slim

# Install system dependencies (ffmpeg is required by Whisper for audio decoding, git is required for installing whisper packages)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies (using PyTorch CPU version to save memory and server costs)
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code and dataset
COPY . .

# Run the FastAPI server, binding to the port injected by the hosting service (defaults to 8000)
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
