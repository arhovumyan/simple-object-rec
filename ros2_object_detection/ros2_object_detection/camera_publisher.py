#!/usr/bin/env python3

"""
ROS2 Camera Publisher Node
Publishes camera feed as ROS2 Image messages
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # Declare parameters
        self.declare_parameter('camera_device', 0)
        self.declare_parameter('camera_width', 960)
        self.declare_parameter('camera_height', 540)
        self.declare_parameter('camera_fps', 30)
        self.declare_parameter('frame_id', 'camera_link')
        
        # Get parameters
        self.camera_device = self.get_parameter('camera_device').value
        self.camera_width = self.get_parameter('camera_width').value
        self.camera_height = self.get_parameter('camera_height').value
        self.camera_fps = self.get_parameter('camera_fps').value
        self.frame_id = self.get_parameter('frame_id').value
        
        # Initialize camera
        self.cap = None
        self.bridge = CvBridge()
        
        # QoS profile for image messages
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create publisher
        self.image_publisher = self.create_publisher(
            Image,
            '/camera/image_raw',
            qos_profile
        )
        
        # Initialize camera
        self.init_camera()
        
        # Create timer for publishing frames
        self.timer = self.create_timer(1.0 / self.camera_fps, self.publish_frame)
        
        self.get_logger().info(f"Camera publisher initialized:")
        self.get_logger().info(f"  Device: {self.camera_device}")
        self.get_logger().info(f"  Resolution: {self.camera_width}x{self.camera_height}")
        self.get_logger().info(f"  FPS: {self.camera_fps}")
        self.get_logger().info(f"  Frame ID: {self.frame_id}")
    
    def init_camera(self):
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_device, cv2.CAP_AVFOUNDATION)
            
            if not self.cap.isOpened():
                self.get_logger().error(f"Failed to open camera device {self.camera_device}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.get_logger().info(f"Camera initialized successfully:")
            self.get_logger().info(f"  Actual resolution: {actual_width}x{actual_height}")
            self.get_logger().info(f"  Actual FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error initializing camera: {e}")
            return False
    
    def publish_frame(self):
        """Publish a single frame"""
        if self.cap is None or not self.cap.isOpened():
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Failed to read frame from camera")
                return
            
            # Create header
            header = self.get_clock().now().to_msg()
            header.frame_id = self.frame_id
            
            # Convert to ROS2 image message
            image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            image_msg.header = header
            
            # Publish image
            self.image_publisher.publish(image_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing frame: {e}")
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CameraPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
