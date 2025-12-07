# Object Recognition System

A robust real-time object detection system using YOLO models and MobileNetV3, optimized for Mac with Apple Silicon GPU support.

## Features


- **Enhanced classification** with MobileNetV3
- **Anti-flickering system** with object persistence
- **GPU acceleration** (Apple MPS for Apple Silicon)
- **Comprehensive logging** (text and JSON formats)
- **Stable bounding boxes** that don't flicker or turn on/off

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## How to run 
    python3 OD_with_model_switching.py
        this is for testing different models without an actual target (manneq. and tent)
    python3 OD_async.py
        our main file which tests for both yolov8n and yolov11s
    OD.py
        intial od file. 
        
