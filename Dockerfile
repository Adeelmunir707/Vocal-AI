# Use the official Python 3.11 image (based on Debian/Ubuntu)
FROM python:3.11-slim

# Install system dependencies (including PortAudio for sounddevice)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    build-essential \
    git \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy application files into the container
COPY . /app/

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port for FastAPI
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
 