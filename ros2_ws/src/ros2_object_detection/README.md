# ROS2 Object Detection Pipeline

A complete ROS2 package for real-time object detection and classification using YOLO and MobileNet models.

## Features

- **Real-time Object Detection**: YOLO-based detection with configurable models (YOLOv8n, YOLOv11s)
- **Object Classification**: MobileNetV3-based classification for target object identification
- **Multi-scale Detection**: Enhanced detection for small objects
- **Image Enhancement**: Advanced preprocessing for improved small object detection
- **Object Tracking**: Hungarian algorithm-based tracking with IoU matching
- **Async Processing**: Queue-based async processing for optimal GPU utilization
- **Motion Detection**: Adaptive detection intervals based on motion
- **ROS2 Integration**: Full ROS2 ecosystem integration with standard message types

## Architecture

```
Camera → Image Publisher → Object Detection Node → Classification Queue → MobileNet → Results
                                    ↓
                              Detection Publisher
                                    ↓
                              Debug Image Publisher
```

## Installation

### Prerequisites

- ROS2 (Humble/Iron/Jazzy)
- Python 3.8+
- OpenCV
- PyTorch
- TensorFlow
- Ultralytics YOLO

### Dependencies

```bash
# Install ROS2 dependencies
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge ros-humble-image-transport

# Install Python dependencies
pip install ultralytics torch tensorflow opencv-python pillow scipy numpy
```

### Build the Package

```bash
# Navigate to your ROS2 workspace
cd ~/ros2_ws/src

# Copy the package
cp -r /path/to/ros2_object_detection .

# Build the package
cd ~/ros2_ws
colcon build --packages-select ros2_object_detection
source install/setup.bash
```

## Usage

### Basic Usage

```bash
# Launch with default parameters
ros2 launch ros2_object_detection object_detection_with_camera.launch.py

# Launch with custom parameters
ros2 launch ros2_object_detection object_detection_with_camera.launch.py \
    yolo_model_path:=yolo11s.pt \
    yolo_confidence:=0.4 \
    enhance_small_objects:=true
```

### Configuration Files

```bash
# Use high performance configuration
ros2 launch ros2_object_detection object_detection.launch.py \
    --params-file config/high_performance.yaml

# Use high accuracy configuration
ros2 launch ros2_object_detection object_detection.launch.py \
    --params-file config/high_accuracy.yaml
```

### External Camera

```bash
# Launch with external camera topic
ros2 launch ros2_object_detection object_detection.launch.py \
    camera_topic:=/your_camera/image_raw
```

## Topics

### Subscribed Topics

- `/camera/image_raw` (sensor_msgs/Image): Input camera feed

### Published Topics

- `/object_detection/detections` (vision_msgs/Detection2DArray): All detected objects
- `/object_detection/targets` (vision_msgs/Detection2DArray): Target objects only
- `/object_detection/debug_image` (sensor_msgs/Image): Debug image with bounding boxes

## Parameters

### Model Parameters

- `yolo_model_path`: Path to YOLO model file (default: "yolo11s.pt")
- `yolo_confidence`: YOLO confidence threshold (default: 0.4)
- `mobilenet_confidence`: MobileNet confidence threshold (default: 0.5)
- `max_detections`: Maximum detections per frame (default: 10)

### Detection Parameters

- `min_box_size`: Minimum bounding box size in pixels (default: 15)
- `multi_scale_detection`: Enable multi-scale detection (default: true)
- `input_resolution`: Input resolution for processing (default: 960)
- `enhance_small_objects`: Enable image enhancement (default: true)
- `ultra_small_detection`: Enable sliding window detection (default: false)

### Tracking Parameters

- `max_track_age`: Maximum track age before removal (default: 8)
- `min_track_hits`: Minimum hits before classification (default: 2)
- `tracking_threshold`: IoU threshold for tracking (default: 0.4)
- `small_object_tracking`: Enable enhanced small object tracking (default: false)

### Camera Parameters

- `camera_device`: Camera device index (default: 0)
- `camera_width`: Camera width in pixels (default: 960)
- `camera_height`: Camera height in pixels (default: 540)
- `camera_fps`: Camera FPS (default: 30)

## Target Objects

The system is configured to detect and classify the following target objects:

- **Tent**: camping_tent, pup_tent, canvas_tent, backpacking_tent, dome_tent
- **Mannequin**: mannequin, dummy, model, display_model, store_dummy, fashion_model

## Performance Optimization

### High Performance Mode

- Higher confidence thresholds
- Lower resolution processing
- Disabled image enhancement
- Optimized tracking parameters
- Larger batch sizes

### High Accuracy Mode

- Lower confidence thresholds
- Higher resolution processing
- Enabled image enhancement
- Enhanced tracking parameters
- Multi-scale detection

## Debugging

### View Topics

```bash
# View detection results
ros2 topic echo /object_detection/detections

# View target objects
ros2 topic echo /object_detection/targets

# View debug image
ros2 run rqt_image_view rqt_image_view /object_detection/debug_image
```

### Monitor Performance

```bash
# Check node status
ros2 node list
ros2 node info /object_detection_pipeline

# Monitor topics
ros2 topic hz /object_detection/detections
ros2 topic bw /camera/image_raw
```

## Troubleshooting

### Common Issues

1. **Camera not found**: Check camera device index and permissions
2. **Low FPS**: Reduce resolution or enable high performance mode
3. **No detections**: Lower confidence thresholds or check lighting
4. **GPU not detected**: Install proper GPU drivers and CUDA/MPS

### Logs

Check the logs directory for detailed information:
```bash
ls logs/
tail -f logs/ros2_object_detection_*.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Ultralytics for YOLO models
- TensorFlow team for MobileNet
- OpenCV community
- ROS2 community
