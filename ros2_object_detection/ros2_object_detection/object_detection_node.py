#!/usr/bin/env python3

"""
ROS2 Object Detection and Classification Pipeline
Converted from OD_async.py to work with ROS2 ecosystem
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
import time
import logging
from datetime import datetime
import os
from queue import Queue, Empty
from threading import Thread, Lock
from collections import deque
import uuid
from scipy.optimize import linear_sum_assignment

# ROS2 imports
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge

# Local imports
from .detection_methods import DetectionMethods

class ROS2ObjectDetectionPipeline(Node):
    def __init__(self):
        super().__init__('object_detection_pipeline')
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("ROS2 OBJECT DETECTION & CLASSIFICATION PIPELINE")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Architecture: Queue-based Async Processing")
        self.get_logger().info("Stage 1: YOLO Object Detection (GPU)")
        self.get_logger().info("Stage 2: MobileNetV3 Classification (GPU)")
        self.get_logger().info("Processing: Async Queue with Batch Processing")
        self.get_logger().info("=" * 60)
        
        # Declare parameters
        self.declare_parameters()
        
        # Setup logging
        self.setup_logging()
        
        # Setup GPU
        self.setup_gpu()
        
        # Model selection
        self.selected_model = self.select_yolo_model()
        
        self.get_logger().info(f"Loading {self.selected_model} model...")
        self.yolo_model = YOLO(self.selected_model)
        if self.device == 'mps':
            self.yolo_model.to(self.device)
        
        self.get_logger().info("Loading MobileNetV3 model...")
        # Enable GPU for TensorFlow/MobileNet
        self.setup_tensorflow_gpu()
        self.mobilenet_model = MobileNetV3Small(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=True
        )
        
        # Queue system for async processing
        self.classification_queue = Queue(maxsize=self.get_parameter('max_queue_size').value)
        self.results_queue = Queue(maxsize=self.get_parameter('max_queue_size').value * 2)
        self.batch_size = self.get_parameter('batch_size').value
        
        # Thread safety
        self.results_lock = Lock()
        self.active_detections = {}  # Track active detections by ID
        
        # Processing control
        self.running = False
        self.classifier_thread = None
        
        # Target objects we're looking for
        self.target_objects = {
            'tent': ['tent', 'camping_tent', 'pup_tent', 'canvas_tent', 'backpacking_tent', 'dome_tent'],
            'mannequin': ['mannequin', 'dummy', 'model', 'display_model', 'store_dummy', 'fashion_model']
        }
        
        # Detection settings - optimized for high FPS
        self.yolo_confidence = self.get_parameter('yolo_confidence').value
        self.mobilenet_confidence = self.get_parameter('mobilenet_confidence').value
        self.max_detections = self.get_parameter('max_detections').value
        
        # Small object detection settings - optimized for high FPS
        self.min_box_size = self.get_parameter('min_box_size').value
        self.multi_scale_detection = self.get_parameter('multi_scale_detection').value
        self.input_resolution = self.get_parameter('input_resolution').value
        self.enhance_small_objects = self.get_parameter('enhance_small_objects').value
        self.ultra_small_detection = self.get_parameter('ultra_small_detection').value
        
        # Adaptive detection settings - optimized for high FPS
        self.motion_threshold = self.get_parameter('motion_threshold').value
        self.static_frame_count = 0   # Count of static frames
        self.motion_detected = False  # Current motion state
        self.base_detection_interval = self.get_parameter('base_detection_interval').value
        self.max_detection_interval = self.get_parameter('max_detection_interval').value
        self.current_detection_interval = self.base_detection_interval
        
        # Motion detection components
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)  # Keep motion history for smoothing
        
        # Object tracking system - optimized for high FPS
        self.tracked_objects = {}  # track_id -> object_info
        self.next_track_id = 1
        self.max_track_age = self.get_parameter('max_track_age').value
        self.min_track_hits = self.get_parameter('min_track_hits').value
        self.tracking_threshold = self.get_parameter('tracking_threshold').value
        self.small_object_tracking = self.get_parameter('small_object_tracking').value
        
        # Statistics
        self.total_detections = 0
        self.target_objects_found = 0
        self.classifications_performed = 0
        self.target_confirmations = 0
        self.queue_stats = {
            'max_queue_size': 0,
            'total_queued': 0,
            'total_processed': 0,
            'avg_queue_time': 0
        }
        
        # Initialize detection methods
        self.detection_methods = DetectionMethods(
            logger=self.logger,
            yolo_model=self.yolo_model,
            mobilenet_model=self.mobilenet_model,
            device=self.device,
            parameters={
                'yolo_confidence': self.yolo_confidence,
                'mobilenet_confidence': self.mobilenet_confidence,
                'max_detections': self.max_detections,
                'min_box_size': self.min_box_size,
                'multi_scale_detection': self.multi_scale_detection,
                'enhance_small_objects': self.enhance_small_objects,
                'ultra_small_detection': self.ultra_small_detection,
                'motion_threshold': self.motion_threshold,
                'base_detection_interval': self.base_detection_interval,
                'max_detection_interval': self.max_detection_interval,
                'max_track_age': self.max_track_age,
                'min_track_hits': self.min_track_hits,
                'tracking_threshold': self.tracking_threshold,
                'small_object_tracking': self.small_object_tracking,
            }
        )
        
        # ROS2 setup
        self.setup_ros2()
        
        self.get_logger().info("Models loaded successfully!")
        self.get_logger().info(f"YOLO Model: {self.selected_model}")
        self.get_logger().info(f"YOLO Device: {self.device.upper()}")
        self.get_logger().info(f"TensorFlow/MobileNet Device: GPU Enabled")
        self.get_logger().info(f"Queue max size: {self.get_parameter('max_queue_size').value}, Batch size: {self.get_parameter('batch_size').value}")
        self.get_logger().info(f"Target objects: {list(self.target_objects.keys())}")
        self.get_logger().info(f"High FPS optimized detection: ENABLED")
        self.get_logger().info(f"Minimum box size: {self.min_box_size}px")
        self.get_logger().info(f"Multi-scale factors: 2 scales (1.0x, 1.5x)")
        self.get_logger().info(f"Sliding window detection: DISABLED (for performance)")
        self.get_logger().info(f"Image enhancement: {'ENABLED' if self.enhance_small_objects else 'DISABLED'}")
        self.get_logger().info(f"Fast tracking: ENABLED")
    
    def declare_parameters(self):
        """Declare ROS2 parameters"""
        # Model parameters
        self.declare_parameter('yolo_model_path', 'yolo11s.pt')
        self.declare_parameter('yolo_confidence', 0.4)
        self.declare_parameter('mobilenet_confidence', 0.5)
        self.declare_parameter('max_detections', 10)
        
        # Detection parameters
        self.declare_parameter('min_box_size', 15)
        self.declare_parameter('multi_scale_detection', True)
        self.declare_parameter('input_resolution', 960)
        self.declare_parameter('enhance_small_objects', True)
        self.declare_parameter('ultra_small_detection', False)
        
        # Motion detection parameters
        self.declare_parameter('motion_threshold', 1000)
        self.declare_parameter('base_detection_interval', 0.2)
        self.declare_parameter('max_detection_interval', 1.0)
        
        # Tracking parameters
        self.declare_parameter('max_track_age', 8)
        self.declare_parameter('min_track_hits', 2)
        self.declare_parameter('tracking_threshold', 0.4)
        self.declare_parameter('small_object_tracking', False)
        
        # Queue parameters
        self.declare_parameter('max_queue_size', 30)
        self.declare_parameter('batch_size', 4)
        
        # ROS2 parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('detections_topic', '/object_detection/detections')
        self.declare_parameter('targets_topic', '/object_detection/targets')
        self.declare_parameter('debug_image_topic', '/object_detection/debug_image')
        self.declare_parameter('frame_id', 'camera_link')
    
    def setup_ros2(self):
        """Setup ROS2 publishers and subscribers"""
        # QoS profile for image messages
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.image_subscription = self.create_subscription(
            Image,
            self.get_parameter('camera_topic').value,
            self.image_callback,
            qos_profile
        )
        
        # Publishers
        self.detections_publisher = self.create_publisher(
            Detection2DArray,
            self.get_parameter('detections_topic').value,
            10
        )
        
        self.targets_publisher = self.create_publisher(
            Detection2DArray,
            self.get_parameter('targets_topic').value,
            10
        )
        
        self.debug_image_publisher = self.create_publisher(
            Image,
            self.get_parameter('debug_image_topic').value,
            10
        )
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Start classification worker thread
        self.running = True
        self.classifier_thread = Thread(target=self.classification_worker, daemon=True)
        self.classifier_thread.start()
        
        self.get_logger().info("ROS2 setup complete!")
    
    def select_yolo_model(self):
        """Allow user to select YOLO model"""
        available_models = {
            '1': ('yolov11s', 'yolo11s.pt', 'YOLOv11 Small - Latest generation, balanced performance'),
            '2': ('yolov8n', 'yolov8n.pt', 'YOLOv8 Nano - Fast, lightweight model')
        }
        
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("YOLO MODEL SELECTION")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Choose your YOLO model:")
        self.get_logger().info("1. YOLOv11 Small (yolo11s.pt) - Latest generation (DEFAULT)")
        self.get_logger().info("   - Newer architecture, better accuracy")
        self.get_logger().info("   - Good balance of speed and performance")
        self.get_logger().info("   - Improved small object detection")
        self.get_logger().info("")
        self.get_logger().info("2. YOLOv8 Nano (yolov8n.pt) - Fast, lightweight model")
        self.get_logger().info("   - Fast, lightweight, good for high FPS")
        self.get_logger().info("   - Previous optimized model")
        self.get_logger().info("=" * 60)
        
        # For ROS2, we'll use the parameter or default
        model_path = self.get_parameter('yolo_model_path').value
        self.get_logger().info(f"Using model from parameter: {model_path}")
        return model_path
    
    def setup_gpu(self):
        """Setup GPU configuration for PyTorch (YOLO)"""
        if torch.backends.mps.is_available():
            self.device = 'mps'
            self.get_logger().info(f"✓ PyTorch GPU (MPS) available")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            self.get_logger().info(f"✓ PyTorch GPU (CUDA) available")
        else:
            self.device = 'cpu'
            self.get_logger().info(f"⚠ Using CPU for PyTorch")
    
    def setup_tensorflow_gpu(self):
        """Setup GPU configuration for TensorFlow (MobileNet)"""
        # Try to enable GPU for TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Enable memory growth to avoid taking all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.get_logger().info(f"TensorFlow GPU enabled: {len(gpus)} GPU(s) found")
            except RuntimeError as e:
                self.get_logger().warn(f"TensorFlow GPU setup warning: {e}")
        else:
            # For macOS with MPS, TensorFlow might not detect GPU
            # But we can still try to use Metal Performance Shaders indirectly
            self.get_logger().info(f"ℹ TensorFlow GPU not detected (normal on macOS)")
            self.get_logger().info(f"  MobileNet will use optimized CPU with potential Metal acceleration")
    
    def setup_logging(self):
        """Setup logging system"""
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger = logging.getLogger('ROS2ObjectDetectionPipeline')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(f'logs/ros2_object_detection_{timestamp}.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.get_logger().info(f"Logging to: logs/ros2_object_detection_{timestamp}.log")
    
    def image_callback(self, msg):
        """Callback for incoming image messages"""
        try:
            # Convert ROS2 image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Process the frame
            self.process_frame(frame, msg.header)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def process_frame(self, frame, header):
        """Process a single frame for object detection"""
        current_time = time.time()
        
        # Detect motion and update detection interval
        motion_detected, motion_level = self.detection_methods.detect_motion(frame)
        self.detection_methods.update_detection_interval(motion_detected)
        
        # Run YOLO detection at adaptive interval
        if not hasattr(self, 'last_detection_time') or current_time - self.last_detection_time >= self.detection_methods.current_detection_interval:
            # Enhance frame for small object detection
            enhanced_frame = self.detection_methods.enhance_image_for_small_objects(frame)
            
            # Stage 1: YOLO Detection
            detections = self.detection_methods.detect_objects_yolo(enhanced_frame)
            
            # Update tracks with new detections
            active_track_ids = self.detection_methods.update_tracks(detections, int(current_time * 1000))
            
            # Get tracks that need classification
            tracks_to_classify = self.detection_methods.get_tracks_needing_classification()
            
            # Queue tracks for classification
            for track_id, track_info in tracks_to_classify:
                success = self.detection_methods.queue_for_classification(
                    frame, track_id, track_info, current_time, self.classification_queue
                )
            
            self.total_detections += len(detections)
            self.last_detection_time = current_time
        
        # Get processed classification results
        processed_results = self.detection_methods.get_processed_results(self.results_queue)
        
        # Count newly classified tracks
        for result in processed_results:
            if result['track_id'] in self.detection_methods.tracked_objects:
                if result['is_target']:
                    self.target_objects_found += 1
        
        # Publish detections
        self.publish_detections(header)
        
        # Publish debug image
        self.publish_debug_image(frame, header)
    
    def publish_detections(self, header):
        """Publish detection results as ROS2 messages"""
        # Create detection array for all objects
        all_detections = Detection2DArray()
        all_detections.header = header
        
        # Create detection array for target objects only
        target_detections = Detection2DArray()
        target_detections.header = header
        
        for track_id, track in self.detection_methods.tracked_objects.items():
            x1, y1, x2, y2 = track['bbox']
            
            # Create detection message
            detection = Detection2D()
            detection.header = header
            
            # Set bounding box
            detection.bbox.center.position.x = float((x1 + x2) / 2)
            detection.bbox.center.position.y = float((y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            
            # Set results
            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = track['yolo_class']
            result.hypothesis.score = track['yolo_confidence']
            detection.results.append(result)
            
            # Add to appropriate array
            all_detections.detections.append(detection)
            
            if track.get('is_target', False):
                target_detections.detections.append(detection)
        
        # Publish detections
        self.detections_publisher.publish(all_detections)
        self.targets_publisher.publish(target_detections)
    
    def publish_debug_image(self, frame, header):
        """Publish debug image with bounding boxes"""
        try:
            # Draw tracked objects
            self.detection_methods.draw_tracked_objects(frame, self.detection_methods.tracked_objects)
            
            # Add stats overlay
            self.add_stats_overlay(frame)
            
            # Convert to ROS2 image message
            debug_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            debug_msg.header = header
            
            # Publish debug image
            self.debug_image_publisher.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing debug image: {e}")
    
    def add_stats_overlay(self, frame):
        """Add statistics overlay to debug image"""
        height, width = frame.shape[:2]
        
        # Display stats
        active_tracks = len(self.detection_methods.tracked_objects)
        stats_text = f"Tracks: {active_tracks} | Detections: {self.total_detections} | Targets: {self.target_objects_found}"
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Pipeline info
        model_display_name = "YOLOv11s" if "yolo11s" in self.selected_model else "YOLOv8n"
        cv2.putText(frame, f"ROS2 Pipeline: {model_display_name} → Tracking → MobileNet", 
                   (10, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "GREEN: Target Objects | RED: Non-Targets | ORANGE: Tracking", 
                   (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def classification_worker(self):
        """Background worker thread for processing classification queue"""
        self.logger.info("Classification worker thread started")
        batch_buffer = []
        last_process_time = time.time()
        batch_timeout = 0.02  # 20ms timeout for faster processing
        
        while self.running:
            try:
                # Try to get items from queue
                try:
                    task = self.classification_queue.get(timeout=0.01)
                    batch_buffer.append(task)
                except Empty:
                    pass
                
                current_time = time.time()
                time_since_last_process = current_time - last_process_time
                
                # Process batch if:
                # 1. Buffer is full (reached batch_size)
                # 2. Buffer has items and timeout reached
                # 3. Queue is empty but buffer has items
                should_process = (
                    len(batch_buffer) >= self.batch_size or
                    (len(batch_buffer) > 0 and time_since_last_process >= batch_timeout) or
                    (len(batch_buffer) > 0 and self.classification_queue.empty())
                )
                
                if should_process and batch_buffer:
                    # Process batch
                    results = self.detection_methods.classify_batch(batch_buffer)
                    
                    # Add results to results queue
                    for result in results:
                        self.results_queue.put(result)
                        self.queue_stats['total_processed'] += 1
                    
                    self.logger.info(f"Processed batch of {len(batch_buffer)} objects")
                    batch_buffer.clear()
                    last_process_time = current_time
                
            except Exception as e:
                self.logger.error(f"Classification worker error: {e}")
                batch_buffer.clear()
        
        self.logger.info("Classification worker thread stopped")
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        self.running = False
        if self.classifier_thread:
            self.classifier_thread.join(timeout=2.0)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ROS2ObjectDetectionPipeline()
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
