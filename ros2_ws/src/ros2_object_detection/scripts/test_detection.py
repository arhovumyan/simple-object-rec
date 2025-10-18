#!/usr/bin/env python3

"""
Test script for ROS2 Object Detection Pipeline
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import time

class DetectionTester(Node):
    def __init__(self):
        super().__init__('detection_tester')
        
        self.bridge = CvBridge()
        self.detection_count = 0
        self.target_count = 0
        
        # Subscribers
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            '/object_detection/detections',
            self.detections_callback,
            10
        )
        
        self.targets_sub = self.create_subscription(
            Detection2DArray,
            '/object_detection/targets',
            self.targets_callback,
            10
        )
        
        self.debug_image_sub = self.create_subscription(
            Image,
            '/object_detection/debug_image',
            self.debug_image_callback,
            10
        )
        
        # Timer for statistics
        self.timer = self.create_timer(5.0, self.print_statistics)
        
        self.get_logger().info("Detection tester started")
        self.get_logger().info("Subscribing to:")
        self.get_logger().info("  /object_detection/detections")
        self.get_logger().info("  /object_detection/targets")
        self.get_logger().info("  /object_detection/debug_image")
    
    def detections_callback(self, msg):
        """Callback for all detections"""
        self.detection_count += len(msg.detections)
        
        if len(msg.detections) > 0:
            self.get_logger().info(f"Received {len(msg.detections)} detections")
            for i, detection in enumerate(msg.detections):
                bbox = detection.bbox
                result = detection.results[0] if detection.results else None
                if result:
                    self.get_logger().info(f"  Detection {i+1}: {result.hypothesis.class_id} "
                                         f"(confidence: {result.hypothesis.score:.2f}) "
                                         f"at ({bbox.center.position.x:.1f}, {bbox.center.position.y:.1f})")
    
    def targets_callback(self, msg):
        """Callback for target detections"""
        self.target_count += len(msg.detections)
        
        if len(msg.detections) > 0:
            self.get_logger().info(f"🎯 TARGET DETECTED! {len(msg.detections)} target(s) found")
            for i, detection in enumerate(msg.detections):
                bbox = detection.bbox
                result = detection.results[0] if detection.results else None
                if result:
                    self.get_logger().info(f"  Target {i+1}: {result.hypothesis.class_id} "
                                         f"(confidence: {result.hypothesis.score:.2f}) "
                                         f"at ({bbox.center.position.x:.1f}, {bbox.center.position.y:.1f})")
    
    def debug_image_callback(self, msg):
        """Callback for debug image"""
        try:
            # Convert ROS2 image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Display the image
            cv2.imshow('ROS2 Object Detection Debug', frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error processing debug image: {e}")
    
    def print_statistics(self):
        """Print detection statistics"""
        self.get_logger().info(f"📊 Statistics (last 5 seconds):")
        self.get_logger().info(f"  Total detections: {self.detection_count}")
        self.get_logger().info(f"  Target detections: {self.target_count}")
        
        # Reset counters
        self.detection_count = 0
        self.target_count = 0

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DetectionTester()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
