#!/usr/bin/env python3

"""
Launch file for ROS2 Object Detection Pipeline
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition
import os

def generate_launch_description():
    """Generate launch description for object detection pipeline"""
    
    # Declare launch arguments
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
    
    mobilenet_confidence_arg = DeclareLaunchArgument(
        'mobilenet_confidence',
        default_value='0.5',
        description='MobileNet confidence threshold'
    )
    
    max_detections_arg = DeclareLaunchArgument(
        'max_detections',
        default_value='10',
        description='Maximum number of detections per frame'
    )
    
    min_box_size_arg = DeclareLaunchArgument(
        'min_box_size',
        default_value='15',
        description='Minimum bounding box size in pixels'
    )
    
    enhance_small_objects_arg = DeclareLaunchArgument(
        'enhance_small_objects',
        default_value='true',
        description='Enable image enhancement for small objects'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Camera image topic'
    )
    
    detections_topic_arg = DeclareLaunchArgument(
        'detections_topic',
        default_value='/object_detection/detections',
        description='Topic for publishing all detections'
    )
    
    targets_topic_arg = DeclareLaunchArgument(
        'targets_topic',
        default_value='/object_detection/targets',
        description='Topic for publishing target detections only'
    )
    
    debug_image_topic_arg = DeclareLaunchArgument(
        'debug_image_topic',
        default_value='/object_detection/debug_image',
        description='Topic for publishing debug image with bounding boxes'
    )
    
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='camera_link',
        description='Frame ID for published messages'
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
            'mobilenet_confidence': LaunchConfiguration('mobilenet_confidence'),
            'max_detections': LaunchConfiguration('max_detections'),
            'min_box_size': LaunchConfiguration('min_box_size'),
            'enhance_small_objects': LaunchConfiguration('enhance_small_objects'),
            'camera_topic': LaunchConfiguration('camera_topic'),
            'detections_topic': LaunchConfiguration('detections_topic'),
            'targets_topic': LaunchConfiguration('targets_topic'),
            'debug_image_topic': LaunchConfiguration('debug_image_topic'),
            'frame_id': LaunchConfiguration('frame_id'),
        }],
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('camera_topic')),
        ]
    )
    
    # Log launch info
    launch_info = LogInfo(
        msg=[
            'Starting ROS2 Object Detection Pipeline with parameters:\n',
            '  YOLO Model: ', LaunchConfiguration('yolo_model_path'), '\n',
            '  YOLO Confidence: ', LaunchConfiguration('yolo_confidence'), '\n',
            '  MobileNet Confidence: ', LaunchConfiguration('mobilenet_confidence'), '\n',
            '  Max Detections: ', LaunchConfiguration('max_detections'), '\n',
            '  Min Box Size: ', LaunchConfiguration('min_box_size'), '\n',
            '  Enhance Small Objects: ', LaunchConfiguration('enhance_small_objects'), '\n',
            '  Camera Topic: ', LaunchConfiguration('camera_topic'), '\n',
            '  Detections Topic: ', LaunchConfiguration('detections_topic'), '\n',
            '  Targets Topic: ', LaunchConfiguration('targets_topic'), '\n',
            '  Debug Image Topic: ', LaunchConfiguration('debug_image_topic'), '\n',
            '  Frame ID: ', LaunchConfiguration('frame_id'), '\n'
        ]
    )
    
    return LaunchDescription([
        yolo_model_arg,
        yolo_confidence_arg,
        mobilenet_confidence_arg,
        max_detections_arg,
        min_box_size_arg,
        enhance_small_objects_arg,
        camera_topic_arg,
        detections_topic_arg,
        targets_topic_arg,
        debug_image_topic_arg,
        frame_id_arg,
        launch_info,
        object_detection_node,
    ])
