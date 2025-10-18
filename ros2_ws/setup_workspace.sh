#!/bin/bash

# ROS2 Object Detection Workspace Setup Script
# This script sets up the ROS2 workspace in the current directory

echo "=========================================="
echo "ROS2 Object Detection Workspace Setup"
echo "=========================================="

# Check if ROS2 is installed
if ! command -v colcon &> /dev/null; then
    echo "❌ ROS2 is not installed or not sourced."
    echo ""
    echo "To install ROS2, please follow these steps:"
    echo ""
    echo "1. Install ROS2 (choose your distribution):"
    echo "   Ubuntu 22.04: sudo apt install ros-humble-desktop"
    echo "   Ubuntu 20.04: sudo apt install ros-foxy-desktop"
    echo "   macOS: brew install ros2"
    echo ""
    echo "2. Source ROS2:"
    echo "   source /opt/ros/humble/setup.bash  # for Humble"
    echo "   source /opt/ros/foxy/setup.bash    # for Foxy"
    echo ""
    echo "3. Install colcon:"
    echo "   sudo apt install python3-colcon-common-extensions"
    echo ""
    echo "4. Then run this script again."
    echo ""
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "src" ]; then
    echo "❌ Please run this script from the ros2_ws directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: ros2_ws directory with src/ subdirectory"
    exit 1
fi

echo "✅ ROS2 found: $(ros2 --version)"
echo "✅ Colcon found: $(colcon --version)"
echo ""

# Install dependencies
echo "📦 Installing ROS2 dependencies..."
sudo apt update
sudo apt install -y \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    python3-colcon-common-extensions

echo ""

# Build the workspace
echo "🔨 Building ROS2 workspace..."
colcon build --packages-select ros2_object_detection

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "🚀 To use the workspace, run:"
    echo "   source install/setup.bash"
    echo ""
    echo "📋 Available launch files:"
    echo "   ros2 launch ros2_object_detection object_detection_with_camera.launch.py"
    echo "   ros2 launch ros2_object_detection object_detection.launch.py"
    echo ""
    echo "🧪 Test the system:"
    echo "   ros2 run ros2_object_detection test_detection.py"
    echo ""
else
    echo "❌ Build failed. Please check the error messages above."
    exit 1
fi
