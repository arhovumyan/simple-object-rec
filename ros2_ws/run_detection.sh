#!/bin/bash

# ROS2 Object Detection Runner Script
# This script runs the object detection pipeline

echo "=========================================="
echo "ROS2 Object Detection Pipeline"
echo "=========================================="

# Check if workspace is built
if [ ! -d "install" ]; then
    echo "❌ Workspace not built. Please run setup_workspace.sh first."
    exit 1
fi

# Source the workspace
echo "🔧 Sourcing workspace..."
source install/setup.bash

# Check if ROS2 is sourced
if ! command -v ros2 &> /dev/null; then
    echo "❌ ROS2 not sourced. Please run:"
    echo "   source /opt/ros/humble/setup.bash  # or your ROS2 distribution"
    exit 1
fi

echo "✅ Workspace sourced successfully"
echo ""

# Show available options
echo "🚀 Available launch options:"
echo "1. Object Detection with Camera (recommended)"
echo "2. Object Detection (external camera)"
echo "3. High Performance Mode"
echo "4. High Accuracy Mode"
echo "5. Test Detection System"
echo ""

read -p "Choose an option (1-5): " choice

case $choice in
    1)
        echo "🎥 Launching Object Detection with Camera..."
        ros2 launch ros2_object_detection object_detection_with_camera.launch.py
        ;;
    2)
        echo "📷 Launching Object Detection (external camera)..."
        ros2 launch ros2_object_detection object_detection.launch.py
        ;;
    3)
        echo "⚡ Launching High Performance Mode..."
        ros2 launch ros2_object_detection object_detection_with_camera.launch.py \
            --params-file src/ros2_object_detection/config/high_performance.yaml
        ;;
    4)
        echo "🎯 Launching High Accuracy Mode..."
        ros2 launch ros2_object_detection object_detection_with_camera.launch.py \
            --params-file src/ros2_object_detection/config/high_accuracy.yaml
        ;;
    5)
        echo "🧪 Running Detection Test..."
        ros2 run ros2_object_detection test_detection.py
        ;;
    *)
        echo "❌ Invalid option. Please choose 1-5."
        exit 1
        ;;
esac
