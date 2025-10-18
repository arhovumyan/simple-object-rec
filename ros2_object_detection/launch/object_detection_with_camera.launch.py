#!/usr/bin/env python3

"""
Launch file for ROS2 Object Detection Pipeline with camera publisher
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    """Generate launch description for object detection pipeline with camera"""
    
    # Declare launch arguments
    camera_device_arg = DeclareLaunchArgument(
        'camera_device',
        default_value='0',
        description='Camera device index (0 for default camera)'
    )
    
    camera_width_arg = DeclareLaunchArgument(
        'camera_width',
        default_value='960',
        description='Camera width in pixels'
    )
    
    camera_height_arg = DeclareLaunchArgument(
        'camera_height',
        default_value='540',
        description='Camera height in pixels'
    )
    
    camera_fps_arg = DeclareLaunchArgument(
        'camera_fps',
        default_value='30',
        description='Camera FPS'
    )
    
    yolo_model_arg = DeclareLaunchArgument(
        'yolo_model_path',
        default_value='yolo11s.pt',
        description='Path to YOLO model file'
    )
    
    yolo_confidence_arg = DeclareLaunchArgument(
        'yolo_confidence',
        default_value='0.4',
        description='YOLO confidence threshold'
    )
    
    enhance_small_objects_arg = DeclareLaunchArgument(
        'enhance_small_objects',
        default_value='true',
        description='Enable image enhancement for small objects'
    )
    
    # Camera publisher node
    camera_publisher_node = Node(
        package='ros2_object_detection',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen',
        parameters=[{
            'camera_device': LaunchConfiguration('camera_device'),
            'camera_width': LaunchConfiguration('camera_width'),
            'camera_height': LaunchConfiguration('camera_height'),
            'camera_fps': LaunchConfiguration('camera_fps'),
        }],
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),
        ]
    )
    
    # Object detection node
    object_detection_node = Node(
        package='ros2_object_detection',
        executable='object_detection_node',
        name='object_detection_pipeline',
        output='screen',
        parameters=[{
            'yolo_model_path': LaunchConfiguration('yolo_model_path'),
            'yolo_confidence': LaunchConfiguration('yolo_confidence'),
            'enhance_small_objects': LaunchConfiguration('enhance_small_objects'),
            'camera_topic': '/camera/image_raw',
            'detections_topic': '/object_detection/detections',
            'targets_topic': '/object_detection/targets',
            'debug_image_topic': '/object_detection/debug_image',
            'frame_id': 'camera_link',
        }]
    )
    
    # Log launch info
    launch_info = LogInfo(
        msg=[
            'Starting ROS2 Object Detection Pipeline with Camera:\n',
            '  Camera Device: ', LaunchConfiguration('camera_device'), '\n',
            '  Camera Resolution: ', LaunchConfiguration('camera_width'), 'x', LaunchConfiguration('camera_height'), '\n',
            '  Camera FPS: ', LaunchConfiguration('camera_fps'), '\n',
            '  YOLO Model: ', LaunchConfiguration('yolo_model_path'), '\n',
            '  YOLO Confidence: ', LaunchConfiguration('yolo_confidence'), '\n',
            '  Enhance Small Objects: ', LaunchConfiguration('enhance_small_objects'), '\n'
        ]
    )
    
    return LaunchDescription([
        camera_device_arg,
        camera_width_arg,
        camera_height_arg,
        camera_fps_arg,
        yolo_model_arg,
        yolo_confidence_arg,
        enhance_small_objects_arg,
        launch_info,
        camera_publisher_node,
        object_detection_node,
    ])
