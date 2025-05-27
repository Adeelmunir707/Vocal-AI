# Use a base image with Python (you can replace this with a different version if needed)
FROM python:3.9-slim

# Install git (needed to clone repositories)
RUN apt-get update && apt-get install -y git

# Set work directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Expose necessary ports (if needed)
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]
