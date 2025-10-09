# Object Recognition System

A robust real-time object detection system using YOLOv8 and MobileNetV3, optimized for Mac with Apple Silicon GPU support.

## Features

- **Real-time object detection** with YOLOv8
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

### 2. Run Detection System
```bash
# Simple way (recommended)
python run_detection.py

# Or directly
source venv/bin/activate
python robust_logged_detection.py
```

### 3. Usage
- **Press 'q'** to quit the detection window
- **Grant camera permissions** when prompted (System Preferences > Security & Privacy > Camera)
- **Gray boxes** show detected objects (no flickering!)

## System Requirements

- **macOS** (tested on Apple Silicon M2)
- **Python 3.8+**
- **Camera access** (built-in or USB camera)
- **8GB+ RAM** recommended

## How It Works

1. **YOLOv8** detects objects in real-time
2. **MobileNetV3** provides additional classification
3. **Object persistence** prevents flickering by maintaining objects for 10 frames
4. **Coordinate smoothing** reduces jittery movement
5. **Confidence smoothing** prevents on/off behavior

## File Structure

```
Object_Recognition/
├── robust_logged_detection.py    # Main detection system
├── run_detection.py              # Simple runner script
├── requirements.txt              # Python dependencies
├── yolov8n.pt                    # YOLOv8 nano model
├── venv/                         # Virtual environment
└── README.md                     # This file
```

## Logs

The system creates detailed logs in the `logs/` directory:
- **Text logs**: Human-readable detection information
- **JSON logs**: Machine-readable data for analysis

## Troubleshooting

### Camera Issues
- Grant camera permissions in System Preferences
- Try different camera backends if needed
- Check camera is not being used by another app

### Performance Issues
- The system automatically uses GPU (MPS) on Apple Silicon
- Lower resolution if needed by modifying the code
- Close other applications to free up resources

### Dependencies
- Make sure all packages are installed: `pip install -r requirements.txt`
- Virtual environment should be activated: `source venv/bin/activate`

## Technical Details

- **YOLOv8n**: Fast, lightweight object detection
- **MobileNetV3**: Efficient image classification
- **OpenCV**: Camera capture and display
- **PyTorch**: Model inference with MPS support
- **Object persistence**: 10-frame tracking system
- **Anti-flickering**: Coordinate and confidence smoothing

## License

This project is for educational and research purposes.