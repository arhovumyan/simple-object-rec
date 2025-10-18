# ROS2 Object Detection Workspace

This workspace contains the ROS2 Object Detection Pipeline package in your local directory.

## 📁 Workspace Structure

```
ros2_ws/
├── src/
│   └── ros2_object_detection/     # The main package
├── build/                         # Build artifacts (created after build)
├── install/                       # Installed packages (created after build)
├── log/                          # Build logs (created after build)
├── setup_workspace.sh            # Setup script
├── run_detection.sh              # Runner script
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Setup the Workspace

```bash
# Make sure you're in the ros2_ws directory
cd /Users/aro/Documents/Object_Recognition/ros2_ws

# Run the setup script
./setup_workspace.sh
```

### 2. Run the Detection Pipeline

```bash
# Use the interactive runner
./run_detection.sh

# Or run directly
source install/setup.bash
ros2 launch ros2_object_detection object_detection_with_camera.launch.py
```

## 📋 Manual Setup (if needed)

### Prerequisites

1. **Install ROS2** (if not already installed):
   ```bash
   # Ubuntu 22.04
   sudo apt install ros-humble-desktop
   
   # Ubuntu 20.04
   sudo apt install ros-foxy-desktop
   
   # macOS
   brew install ros2
   ```

2. **Source ROS2**:
   ```bash
   # For Humble
   source /opt/ros/humble/setup.bash
   
   # For Foxy
   source /opt/ros/foxy/setup.bash
   ```

3. **Install Dependencies**:
   ```bash
   sudo apt install python3-colcon-common-extensions
   sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge ros-humble-image-transport
   ```

### Build the Workspace

```bash
# From the ros2_ws directory
colcon build --packages-select ros2_object_detection
source install/setup.bash
```

## 🎯 Usage Examples

### Basic Usage

```bash
# Launch with camera
ros2 launch ros2_object_detection object_detection_with_camera.launch.py

# Launch with external camera
ros2 launch ros2_object_detection object_detection.launch.py \
    camera_topic:=/your_camera/image_raw
```

### Configuration Modes

```bash
# High Performance Mode
ros2 launch ros2_object_detection object_detection_with_camera.launch.py \
    --params-file src/ros2_object_detection/config/high_performance.yaml

# High Accuracy Mode
ros2 launch ros2_object_detection object_detection_with_camera.launch.py \
    --params-file src/ros2_object_detection/config/high_accuracy.yaml
```

### Custom Parameters

```bash
ros2 launch ros2_object_detection object_detection_with_camera.launch.py \
    yolo_model_path:=yolo11s.pt \
    yolo_confidence:=0.4 \
    enhance_small_objects:=true \
    camera_width:=1280 \
    camera_height:=720
```

## 🧪 Testing

```bash
# Test the detection system
ros2 run ros2_object_detection test_detection.py

# View topics
ros2 topic list
ros2 topic echo /object_detection/detections
ros2 topic echo /object_detection/targets

# View debug image
ros2 run rqt_image_view rqt_image_view /object_detection/debug_image
```

## 📡 Available Topics

### Subscribed Topics
- `/camera/image_raw` - Camera feed

### Published Topics
- `/object_detection/detections` - All detected objects
- `/object_detection/targets` - Target objects only
- `/object_detection/debug_image` - Debug image with bounding boxes

## 🔧 Troubleshooting

### Common Issues

1. **"colcon: command not found"**
   - Install colcon: `sudo apt install python3-colcon-common-extensions`
   - Make sure ROS2 is sourced

2. **"Package not found"**
   - Make sure you're in the ros2_ws directory
   - Run `source install/setup.bash`

3. **Camera not found**
   - Check camera permissions
   - Try different camera device index (0, 1, 2, etc.)

4. **Build errors**
   - Check Python dependencies: `pip install ultralytics torch tensorflow opencv-python`
   - Make sure all ROS2 dependencies are installed

### Logs

Check build logs:
```bash
cat log/latest_build/ros2_object_detection/stdout_stderr.log
```

## 📚 More Information

For detailed information about the package, see:
- `src/ros2_object_detection/README.md` - Package documentation
- `src/ros2_object_detection/config/` - Configuration files
- `src/ros2_object_detection/launch/` - Launch files

## 🎉 Success!

Once everything is working, you should see:
- Camera feed with bounding boxes
- Detection messages in the terminal
- Target object notifications
- Debug image window (if using test script)
