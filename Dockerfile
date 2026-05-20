FROM python:3.10-slim

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies (only lightweight request and web libraries now)
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code and dataset
COPY . .

# Run the FastAPI server
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
