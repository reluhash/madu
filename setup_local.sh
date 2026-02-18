#!/bin/bash

# Update package lists
sudo apt-get update

# Install system dependencies
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install Python dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install transformers paddleocr paddlepaddle opencv-python-headless loguru fastapi uvicorn pydantic

# Clone PaddleOCR if needed for more complex OCR tasks
# git clone https://github.com/PaddlePaddle/PaddleOCR.git

echo "Local environment setup complete. You can now run 'python3 run_modular.py' with your document image."
