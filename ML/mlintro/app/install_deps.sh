#!/bin/bash

# System configuration
set -e

# Install system dependencies
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y \
    ffmpeg \
    python3-pip \
    python3-venv \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    portaudio19-dev \
    python3-dev \
    nvidia-cuda-toolkit

# Create working directory
sudo mkdir -p /var/video_processing
sudo chown $USER:$USER /var/video_processing

# Set up Python environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

# Download NLTK data
python3 -m nltk.downloader punkt averaged_perceptron_tagger wordnet

echo ""
echo "Installation complete!"
echo "1. Start the service: source venv/bin/activate && python main.py"
echo "2. Test with:"
echo "   curl -X POST http://localhost:8000/process_video/ \\"
echo "        -H \"Content-Type: application/json\" \\"
echo "        -d '{\"driveLink\": \"YOUR_DRIVE_LINK\", \"questionType\": \"technical\", \"keywords\": [\"AI\"]}'"