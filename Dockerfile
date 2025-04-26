# Base Python image
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements file
# COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk-3-0 \
    ffmpeg \
    libpq-dev \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install torch torchvision opencv-python pillow ultralytics

# Set display environment variable for GUI
ENV DISPLAY=:0

# Copy application code
COPY . .

# Set default command
CMD ["python", "main.py"]
