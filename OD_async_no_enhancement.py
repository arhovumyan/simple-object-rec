#!/usr/bin/env python3

"""
Object Detection and Classification Pipeline with Queue-based Async Processing
Optimized for GPU efficiency using concurrent processing with queues.

Architecture:
- YOLO runs on every frame (GPU) - Fast detection
- Detected objects queue up for MobileNet classification
- MobileNet processes queue in batches (GPU) - Efficient classification
- Results merge back for display

Benefits:
- Both models run on GPU without contention
- Sequential pipeline with async processing
- Lower overhead than multiprocessing
- Better GPU utilization
"""

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
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

class AsyncObjectDetectionPipeline:
    def __init__(self, max_queue_size=30, batch_size=4):
        print("=" * 60)
        print("ASYNC OBJECT DETECTION & CLASSIFICATION PIPELINE")
        print("=" * 60)
        print("Architecture: Queue-based Async Processing")
        print("Stage 1: YOLO Object Detection (GPU)")
        print("Stage 2: MobileNetV3 Classification (GPU)")
        print("Processing: Async Queue with Batch Processing")
        print("=" * 60)
        
        self.setup_logging()
        
        self.setup_gpu()
        
        # Model selection
        self.selected_model = self.select_yolo_model()
        
        print(f"Loading {self.selected_model} model...")
        self.yolo_model = YOLO(self.selected_model)
        if self.device == 'mps':
            self.yolo_model.to(self.device)
        
        print("Loading MobileNetV3 model...")
        # Enable GPU for TensorFlow/MobileNet
        self.setup_tensorflow_gpu()
        self.mobilenet_model = MobileNetV3Small(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=True
        )
        
        # Queue system for async processing
        self.classification_queue = Queue(maxsize=max_queue_size)
        self.results_queue = Queue(maxsize=max_queue_size * 2)
        self.batch_size = batch_size
        
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
        self.yolo_confidence = 0.4  # Higher confidence for faster processing
        self.mobilenet_confidence = 0.5  # Higher threshold for classification
        self.max_detections = 10  # Lower limit for better performance
        
        # Small object detection settings - optimized for high FPS
        self.min_box_size = 15  # Slightly larger minimum for faster processing
        self.multi_scale_detection = True
        self.input_resolution = 960  # Lower resolution for better FPS
        self.enhance_small_objects = False  # Disable enhancement for performance
        self.ultra_small_detection = False  # Disable sliding window for performance
        
        # Adaptive detection settings - optimized for high FPS
        self.motion_threshold = 1000  # Motion detection threshold
        self.static_frame_count = 0   # Count of static frames
        self.motion_detected = False  # Current motion state
        self.base_detection_interval = 0.2  # 200ms base interval for better FPS
        self.max_detection_interval = 1.0   # 1 second max interval when static
        self.current_detection_interval = self.base_detection_interval
        
        # High-frequency target object detection (10Hz = 100ms)
        self.target_detection_interval = 0.1  # 100ms for 10Hz target detection
        self.last_target_detection_time = 0
        self.active_target_tracks = set()  # Track IDs that are confirmed targets
        
        # Motion detection components
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)  # Keep motion history for smoothing
        
        # Object tracking system - optimized for high FPS
        self.tracked_objects = {}  # track_id -> object_info
        self.next_track_id = 1
        self.max_track_age = 8  # Shorter age for better performance
        self.min_track_hits = 2  # Lower hits threshold for faster tracking
        self.tracking_threshold = 0.4  # Higher IoU threshold for faster processing
        self.small_object_tracking = False  # Disable for better performance
        
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
        
        self.logger.info("Models loaded successfully!")
        self.logger.info(f"YOLO Model: {self.selected_model}")
        self.logger.info(f"YOLO Device: {self.device.upper()}")
        self.logger.info(f"TensorFlow/MobileNet Device: GPU Enabled")
        self.logger.info(f"Queue max size: {max_queue_size}, Batch size: {batch_size}")
        self.logger.info(f"Target objects: {list(self.target_objects.keys())}")
        self.logger.info(f"High FPS optimized detection: ENABLED")
        self.logger.info(f"Minimum box size: {self.min_box_size}px")
        self.logger.info(f"Multi-scale factors: 2 scales (1.0x, 1.5x)")
        self.logger.info(f"Sliding window detection: DISABLED (for performance)")
        self.logger.info(f"Image enhancement: DISABLED (for performance)")
        self.logger.info(f"Fast tracking: ENABLED")
    
    def select_yolo_model(self):
        """Allow user to select YOLO model"""
        available_models = {
            '1': ('yolov11s', 'yolo11s.pt', 'YOLOv11 Small - Latest generation, balanced performance'),
            '2': ('yolov8n', 'yolov8n.pt', 'YOLOv8 Nano - Fast, lightweight model')
        }
        
        print("\n" + "=" * 60)
        print("YOLO MODEL SELECTION")
        print("=" * 60)
        print("Choose your YOLO model:")
        print("1. YOLOv11 Small (yolo11s.pt) - Latest generation (DEFAULT)")
        print("   - Newer architecture, better accuracy")
        print("   - Good balance of speed and performance")
        print("   - Improved small object detection")
        print()
        print("2. YOLOv8 Nano (yolov8n.pt) - Fast, lightweight model")
        print("   - Fast, lightweight, good for high FPS")
        print("   - Previous optimized model")
        print("=" * 60)
        
        while True:
            try:
                choice = input("Enter your choice (1 or 2): ").strip()
                if choice in available_models:
                    model_name, model_path, description = available_models[choice]
                    print(f"\n✓ Selected: {model_name}")
                    print(f"  {description}")
                    print("=" * 60)
                    return model_path
                else:
                    print("Please enter 1 or 2")
            except KeyboardInterrupt:
                print("\nUsing default model (yolo11s.pt)")
                return 'yolo11s.pt'
    
    def setup_gpu(self):
        """Setup GPU configuration for PyTorch (YOLO)"""
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print(f"✓ PyTorch GPU (MPS) available")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print(f"✓ PyTorch GPU (CUDA) available")
        else:
            self.device = 'cpu'
            print(f"⚠ Using CPU for PyTorch")
    
    def setup_tensorflow_gpu(self):
        """Setup GPU configuration for TensorFlow (MobileNet)"""
        # Try to enable GPU for TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Enable memory growth to avoid taking all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"TensorFlow GPU enabled: {len(gpus)} GPU(s) found")
                self.logger.info(f"TensorFlow GPUs: {gpus}")
            except RuntimeError as e:
                print(f"TensorFlow GPU setup warning: {e}")
        else:
            # For macOS with MPS, TensorFlow might not detect GPU
            # But we can still try to use Metal Performance Shaders indirectly
            print(f"ℹ TensorFlow GPU not detected (normal on macOS)")
            print(f"  MobileNet will use optimized CPU with potential Metal acceleration")
    
    def setup_logging(self):
        """Setup logging system"""
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger = logging.getLogger('AsyncObjectDetectionPipeline')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(f'logs/async_object_detection_{timestamp}.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        print(f"Logging to: logs/async_object_detection_{timestamp}.log")
    
    def detect_objects_yolo(self, frame):
        """Stage 1: Use YOLOv8 for object detection with ultra-aggressive multi-scale support"""
        try:
            # Ultra-aggressive multi-scale detection for smallest objects
            all_detections = []
            
            if self.multi_scale_detection:
                height, width = frame.shape[:2]
                
                # Scale factors for different object sizes - optimized for high FPS
                scale_factors = [1.0, 1.5]  # Only 2 scales for maximum performance
                confidence_factors = [1.0, 0.8]  # Decreasing confidence for higher scales
                
                for i, (scale, conf_factor) in enumerate(zip(scale_factors, confidence_factors)):
                    if scale == 1.0:
                        # Original scale
                        results = self.yolo_model(
                            frame,
                            conf=self.yolo_confidence * conf_factor,
                            iou=0.5,  # Standard IoU for better performance
                            max_det=self.max_detections,
                            verbose=False,
                            device=self.device
                        )
                        processed_frame = frame
                    else:
                        # Scaled detection
                        scaled_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                        results = self.yolo_model(
                            scaled_frame,
                            conf=self.yolo_confidence * conf_factor,
                            iou=0.5,  # Standard IoU for better performance
                            max_det=self.max_detections,
                            verbose=False,
                            device=self.device
                        )
                        processed_frame = scaled_frame
                    
                    if results and len(results) > 0:
                        scale_detections = self._process_yolo_results(results, processed_frame.shape)
                        
                        # Scale coordinates back to original frame if needed
                        if scale != 1.0:
                            for detection in scale_detections:
                                x1, y1, x2, y2 = detection['bbox']
                                detection['bbox'] = (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale))
                                # Mark as scaled detection for priority in merging
                                detection['scale'] = scale
                                # Slightly reduce confidence for scaled detections to prevent duplicates
                                detection['confidence'] *= 0.95
                        
                        all_detections.extend(scale_detections)
                        
                        # Log scale detection results
                        self.logger.debug(f"Scale {scale}x: Found {len(scale_detections)} detections")
                
                # Sliding window detection disabled for performance
                # (Ultra-small detection can be enabled if needed for specific use cases)
                    
            else:
                # Single scale detection
                results = self.yolo_model(
                    frame,
                    conf=self.yolo_confidence,
                    iou=0.5,  # Standard IoU
                    max_det=self.max_detections,
                    verbose=False,
                    device=self.device
                )
                if results and len(results) > 0:
                    all_detections = self._process_yolo_results(results, frame.shape)
            
            # Remove duplicates and merge results with more aggressive merging
            detections = self._merge_detections_aggressive(all_detections)
            
            # Final filtering to keep only reasonable detections
            detections = self._filter_detections_by_size(detections)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
            return []
    
    def _process_yolo_results(self, results, frame_shape):
        """Process YOLO results into detection format"""
        detections = []
        result = results[0]
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                confidence = float(box.conf[0].cpu())
                class_id = int(box.cls[0].cpu())
                class_name = self.yolo_model.names[class_id]
                
                # Only filter extremely small detections
                if (x2 - x1) < self.min_box_size or (y2 - y1) < self.min_box_size:
                    continue
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_name': class_name,
                    'class_id': class_id
                })
        
        return detections
    
    def _detect_ultra_small_objects(self, frame):
        """Detect ultra-small objects using sliding window approach"""
        ultra_small_detections = []
        height, width = frame.shape[:2]
        
        # Sliding window parameters for ultra-small objects
        window_size = 224  # MobileNet input size
        stride = 112  # 50% overlap
        min_confidence = self.yolo_confidence * 0.3  # Very low confidence for ultra-small
        
        try:
            # Slide window across the image
            for y in range(0, height - window_size + 1, stride):
                for x in range(0, width - window_size + 1, stride):
                    # Extract window
                    window = frame[y:y+window_size, x:x+window_size]
                    
                    # Run YOLO on window
                    results = self.yolo_model(
                        window,
                        conf=min_confidence,
                        iou=0.2,  # Very low IoU
                        max_det=10,
                        verbose=False,
                        device=self.device
                    )
                    
                    if results and len(results) > 0:
                        window_detections = self._process_yolo_results(results, window.shape)
                        
                        # Adjust coordinates to original frame
                        for detection in window_detections:
                            wx1, wy1, wx2, wy2 = detection['bbox']
                            # Convert window coordinates to frame coordinates
                            detection['bbox'] = (x + wx1, y + wy1, x + wx2, y + wy2)
                            # Mark as ultra-small detection
                            detection['ultra_small'] = True
                        
                        ultra_small_detections.extend(window_detections)
            
            self.logger.debug(f"Ultra-small detection: Found {len(ultra_small_detections)} detections")
            
        except Exception as e:
            self.logger.error(f"Ultra-small detection error: {e}")
        
        return ultra_small_detections
    
    def _filter_detections_by_size(self, detections):
        """Filter detections to keep only reasonable sizes"""
        filtered = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Keep detections that are at least minimum size
            # but not too large (likely false positives)
            if (width >= self.min_box_size and height >= self.min_box_size and
                width < 800 and height < 800):  # Reasonable upper bound
                filtered.append(detection)
        
        return filtered
    
    def _merge_detections_aggressive(self, all_detections):
        """Enhanced merging of detections with strict overlap prevention"""
        if not all_detections:
            return []
        
        # Sort by confidence (highest first), but prioritize original scale (1.0) detections
        all_detections.sort(key=lambda x: (x.get('scale', 1.0) == 1.0, x['confidence']), reverse=True)
        
        merged_detections = []
        
        for detection in all_detections:
            x1, y1, x2, y2 = detection['bbox']
            is_duplicate = False
            
            # Calculate box area for size-based IoU threshold
            box_area = (x2 - x1) * (y2 - y1)
            
            for merged in merged_detections:
                mx1, my1, mx2, my2 = merged['bbox']
                
                # Calculate IoU
                iou = self.calculate_iou((x1, y1, x2, y2), (mx1, my1, mx2, my2))
                
                # Use stricter IoU thresholds to prevent overlaps
                if box_area < 400:  # Very small objects
                    iou_threshold = 0.2  # Stricter threshold
                elif box_area < 1600:  # Small objects
                    iou_threshold = 0.3  # Stricter threshold
                else:  # Larger objects
                    iou_threshold = 0.4  # Stricter threshold
                
                # Check for overlap regardless of class - prevent any overlapping boxes
                if iou > iou_threshold:
                    # If same class, merge and keep higher confidence
                    if detection['class_name'] == merged['class_name']:
                        if detection['confidence'] > merged['confidence']:
                            merged['bbox'] = detection['bbox']
                            merged['confidence'] = detection['confidence']
                            if 'ultra_small' in detection:
                                merged['ultra_small'] = detection['ultra_small']
                    else:
                        # Different classes but overlapping - keep the one with higher confidence
                        if detection['confidence'] > merged['confidence']:
                            # Replace the merged detection
                            merged_detections.remove(merged)
                            merged_detections.append(detection)
                            merged = detection  # Update reference
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_detections.append(detection)
                
                # Limit total detections for performance
                if len(merged_detections) >= self.max_detections:
                    break
        
        return merged_detections
    
    
    def _merge_detections(self, all_detections):
        """Merge detections from multiple scales and remove duplicates"""
        if not all_detections:
            return []
        
        # Sort by confidence (highest first)
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged_detections = []
        
        for detection in all_detections:
            x1, y1, x2, y2 = detection['bbox']
            is_duplicate = False
            
            for merged in merged_detections:
                mx1, my1, mx2, my2 = merged['bbox']
                
                # Calculate IoU
                iou = self.calculate_iou((x1, y1, x2, y2), (mx1, my1, mx2, my2))
                
                # If IoU is high and same class, consider it a duplicate
                if iou > 0.5 and detection['class_name'] == merged['class_name']:
                    # Keep the one with higher confidence
                    if detection['confidence'] > merged['confidence']:
                        merged['bbox'] = detection['bbox']
                        merged['confidence'] = detection['confidence']
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_detections.append(detection)
                
                # Limit total detections
                if len(merged_detections) >= self.max_detections:
                    break
        
        return merged_detections
    
    def enhance_image_for_small_objects(self, frame):
        """Enhance image to improve small object detection with advanced preprocessing"""
        if not self.enhance_small_objects:
            return frame
        
        try:
            # Create multiple enhanced versions
            enhanced_versions = []
            
            # Version 1: CLAHE enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # More aggressive CLAHE
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced1 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            enhanced_versions.append(enhanced1)
            
            # Version 2: Gamma correction for better contrast
            gamma = 1.2  # Slight gamma boost
            gamma_corrected = np.power(frame / 255.0, gamma) * 255.0
            enhanced2 = np.uint8(gamma_corrected)
            enhanced_versions.append(enhanced2)
            
            # Version 3: Unsharp masking for edge enhancement
            gaussian = cv2.GaussianBlur(frame, (5, 5), 1.0)
            unsharp_mask = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
            enhanced3 = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
            enhanced_versions.append(enhanced3)
            
            # Version 4: Bilateral filter for noise reduction while preserving edges
            bilateral = cv2.bilateralFilter(frame, 9, 75, 75)
            enhanced_versions.append(bilateral)
            
            # Version 5: Morphological operations for small object enhancement
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Top-hat operation to enhance small bright objects
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            tophat_color = cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)
            enhanced5 = cv2.add(frame, tophat_color)
            enhanced_versions.append(enhanced5)
            
            # Combine all enhanced versions using weighted average
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # CLAHE gets highest weight
            result = np.zeros_like(frame, dtype=np.float32)
            
            for enhanced, weight in zip(enhanced_versions, weights):
                result += enhanced.astype(np.float32) * weight
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # Final sharpening for small objects
            kernel_sharp = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(result, -1, kernel_sharp)
            
            # Blend original and final enhanced image
            final_result = cv2.addWeighted(frame, 0.3, sharpened, 0.7, 0)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Advanced image enhancement error: {e}")
            return frame
    
    def deduplicate_detections(self, raw_detections):
        """Remove overlapping detections of the same class"""
        if not raw_detections:
            return []
        
        sorted_detections = sorted(raw_detections, key=lambda x: x['confidence'], reverse=True)
        filtered_detections = []
        
        for detection in sorted_detections:
            x1, y1, x2, y2 = detection['bbox']
            is_duplicate = False
            
            for accepted in filtered_detections:
                ax1, ay1, ax2, ay2 = accepted['bbox']
                
                # Calculate IoU
                intersection_x1 = max(x1, ax1)
                intersection_y1 = max(y1, ay1)
                intersection_x2 = min(x2, ax2)
                intersection_y2 = min(y2, ay2)
                
                if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                    box_area = (x2 - x1) * (y2 - y1)
                    accepted_area = (ax2 - ax1) * (ay2 - ay1)
                    union_area = box_area + accepted_area - intersection_area
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    if iou > 0.3:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_detections.append(detection)
                
                if len(filtered_detections) >= 3:
                    break
        
        return filtered_detections
    
    def queue_for_classification(self, frame, track_id, track_info, timestamp):
        """Queue a tracked object for MobileNet classification"""
        try:
            x1, y1, x2, y2 = track_info['bbox']
            cropped_obj = frame[y1:y2, x1:x2]
            
            if cropped_obj.size == 0:
                return False
            
            # Create classification task
            task = {
                'track_id': track_id,
                'cropped_image': cropped_obj.copy(),
                'bbox': track_info['bbox'],
                'yolo_class': track_info['yolo_class'],
                'yolo_confidence': track_info['yolo_confidence'],
                'timestamp': timestamp,
                'queued_at': time.time()
            }
            
            # Try to add to queue (non-blocking)
            if not self.classification_queue.full():
                self.classification_queue.put(task, block=False)
                self.queue_stats['total_queued'] += 1
                
                # Track queue size
                current_size = self.classification_queue.qsize()
                if current_size > self.queue_stats['max_queue_size']:
                    self.queue_stats['max_queue_size'] = current_size
                
                return True
            else:
                self.logger.warning("Classification queue full, skipping detection")
                return False
                
        except Exception as e:
            self.logger.error(f"Error queueing detection: {e}")
            return False
    
    def classify_batch(self, batch_tasks):
        """Classify a batch of cropped images with MobileNet"""
        if not batch_tasks:
            return []
        
        try:
            # Prepare batch
            batch_images = []
            for task in batch_tasks:
                # Resize and preprocess
                resized_image = cv2.resize(task['cropped_image'], (224, 224))
                rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(rgb_image)
                img_array = image.img_to_array(pil_image)
                img_array = preprocess_input(img_array)
                batch_images.append(img_array)
            
            # Stack into batch - use np.array to preserve shape (batch_size, 224, 224, 3)
            batch_array = np.array(batch_images)
            
            # Run batch prediction on GPU
            predictions = self.mobilenet_model.predict(batch_array, verbose=0)
            
            # Process results
            results = []
            for i, task in enumerate(batch_tasks):
                pred = predictions[i:i+1]
                decoded = decode_predictions(pred, top=3)[0]
                
                if decoded:
                    top_prediction = decoded[0]
                    result = {
                        'track_id': task['track_id'],
                        'bbox': task['bbox'],
                        'yolo_class': task['yolo_class'],
                        'yolo_confidence': task['yolo_confidence'],
                        'mobilenet_class': top_prediction[1],
                        'mobilenet_confidence': top_prediction[2],
                        'all_predictions': decoded,
                        'timestamp': task['timestamp'],
                        'queue_time': time.time() - task['queued_at']
                    }
                    results.append(result)
                    self.classifications_performed += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch classification error: {e}")
            return []
    
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
                    results = self.classify_batch(batch_buffer)
                    
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
    
    def is_target_object(self, class_name):
        """Check if the classified object is one of our target objects"""
        class_lower = class_name.lower()
        
        for target_type, keywords in self.target_objects.items():
            for keyword in keywords:
                if keyword.lower() in class_lower:
                    return True, target_type
        
        return False, None
    
    def detect_motion(self, frame):
        """Detect motion between current and previous frame"""
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return False, 0
        
        try:
            # Convert to grayscale for motion detection
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)
            gray_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)
            
            # Compute frame difference
            frame_diff = cv2.absdiff(gray_current, gray_prev)
            
            # Apply threshold to get binary image
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Count non-zero pixels (motion pixels)
            motion_pixels = cv2.countNonZero(thresh)
            
            # Update motion history
            self.motion_history.append(motion_pixels)
            
            # Calculate average motion over recent frames
            avg_motion = sum(self.motion_history) / len(self.motion_history)
            
            # Determine if motion is detected
            motion_detected = avg_motion > self.motion_threshold
            
            # Update static frame counter
            if not motion_detected:
                self.static_frame_count += 1
            else:
                self.static_frame_count = 0
            
            # Update previous frame
            self.prev_frame = frame.copy()
            
            return motion_detected, avg_motion
            
        except Exception as e:
            self.logger.error(f"Motion detection error: {e}")
            return False, 0
    
    def update_detection_interval(self, motion_detected):
        """Adaptively update detection interval based on motion"""
        if motion_detected:
            # Motion detected - use fast detection
            self.current_detection_interval = self.base_detection_interval
            self.motion_detected = True
        else:
            # No motion - gradually increase interval up to maximum
            if self.static_frame_count > 30:  # After 3 seconds of no motion
                self.current_detection_interval = min(
                    self.current_detection_interval * 1.1,  # Gradually increase
                    self.max_detection_interval
                )
            elif self.static_frame_count > 10:  # After 1 second of no motion
                self.current_detection_interval = min(
                    self.current_detection_interval * 1.05,  # Slowly increase
                    self.max_detection_interval
                )
            
            self.motion_detected = False
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def update_tracks(self, detections, frame_count):
        """Update object tracks with new detections"""
        # Age existing tracks
        for track_id in list(self.tracked_objects.keys()):
            self.tracked_objects[track_id]['age'] += 1
            
            # Remove old tracks
            if self.tracked_objects[track_id]['age'] > self.max_track_age:
                del self.tracked_objects[track_id]
        
        # Create detection matrix for Hungarian algorithm
        if not detections or not self.tracked_objects:
            # No existing tracks, create new ones
            for detection in detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracked_objects[track_id] = {
                    'bbox': detection['bbox'],
                    'yolo_class': detection['class_name'],
                    'yolo_confidence': detection['confidence'],
                    'hits': 1,
                    'age': 0,
                    'last_seen': frame_count,
                    'classified': False,
                    'classification_result': None,
                    'is_target': False,
                    'target_type': None
                }
            return list(self.tracked_objects.keys())
        
        # Calculate cost matrix with enhanced small object support
        detections_len = len(detections)
        tracks_len = len(self.tracked_objects)
        
        if detections_len == 0:
            return list(self.tracked_objects.keys())
        
        cost_matrix = np.zeros((tracks_len, detections_len))
        track_ids = list(self.tracked_objects.keys())
        
        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracked_objects[track_id]['bbox']
            track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
            
            for j, detection in enumerate(detections):
                detection_bbox = detection['bbox']
                detection_area = (detection_bbox[2] - detection_bbox[0]) * (detection_bbox[3] - detection_bbox[1])
                
                # Calculate IoU
                iou = self.calculate_iou(track_bbox, detection_bbox)
                
                # Enhanced cost calculation for small objects
                if self.small_object_tracking:
                    # For small objects, use distance-based cost as well
                    track_center = ((track_bbox[0] + track_bbox[2]) // 2, (track_bbox[1] + track_bbox[3]) // 2)
                    detection_center = ((detection_bbox[0] + detection_bbox[2]) // 2, (detection_bbox[1] + detection_bbox[3]) // 2)
                    
                    # Calculate Euclidean distance
                    distance = np.sqrt((track_center[0] - detection_center[0])**2 + (track_center[1] - detection_center[1])**2)
                    
                    # Normalize distance (max expected distance for small objects)
                    max_distance = 100  # pixels
                    normalized_distance = min(distance / max_distance, 1.0)
                    
                    # Combine IoU and distance costs
                    iou_cost = 1.0 - iou
                    distance_cost = normalized_distance
                    
                    # Weight more heavily on IoU for larger objects, distance for smaller objects
                    if track_area < 400:  # Small objects
                        cost_matrix[i, j] = 0.3 * iou_cost + 0.7 * distance_cost
                    else:  # Larger objects
                        cost_matrix[i, j] = 0.8 * iou_cost + 0.2 * distance_cost
                else:
                    cost_matrix[i, j] = 1.0 - iou  # Standard IoU cost
        
        # Hungarian algorithm for optimal assignment
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Update matched tracks
        matched_tracks = set()
        matched_detections = set()
        
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            track_id = track_ids[track_idx]
            detection = detections[detection_idx]
            
            # Adaptive threshold based on object size
            track_bbox = self.tracked_objects[track_id]['bbox']
            track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
            
            # Use lower threshold for small objects
            if track_area < 400:  # Small objects
                adaptive_threshold = 0.15  # Very low threshold
            elif track_area < 1600:  # Medium objects
                adaptive_threshold = self.tracking_threshold
            else:  # Large objects
                adaptive_threshold = self.tracking_threshold * 1.2
            
            # Only update if cost is below threshold
            if cost_matrix[track_idx, detection_idx] < (1.0 - adaptive_threshold):
                self.tracked_objects[track_id]['bbox'] = detection['bbox']
                self.tracked_objects[track_id]['yolo_class'] = detection['class_name']
                self.tracked_objects[track_id]['yolo_confidence'] = detection['confidence']
                self.tracked_objects[track_id]['hits'] += 1
                self.tracked_objects[track_id]['age'] = 0
                self.tracked_objects[track_id]['last_seen'] = frame_count
                
                # For small objects, give bonus hits for consistent tracking
                if track_area < 400 and self.tracked_objects[track_id]['hits'] > 3:
                    self.tracked_objects[track_id]['hits'] += 1
                
                matched_tracks.add(track_id)
                matched_detections.add(detection_idx)
        
        # Create new tracks for unmatched detections
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracked_objects[track_id] = {
                    'bbox': detection['bbox'],
                    'yolo_class': detection['class_name'],
                    'yolo_confidence': detection['confidence'],
                    'hits': 1,
                    'age': 0,
                    'last_seen': frame_count,
                    'classified': False,
                    'classification_result': None,
                    'is_target': False,
                    'target_type': None
                }
        
        return list(self.tracked_objects.keys())
    
    def update_target_tracks_high_freq(self, detections, frame_count):
        """Update confirmed target tracks with high-frequency detection"""
        if not self.active_target_tracks or not detections:
            return
        
        # Filter detections for target objects only
        target_detections = []
        for detection in detections:
            class_name = detection.get('class_name', '')
            if class_name in ['person', 'backpack', 'suitcase', 'handbag', 'tie']:
                target_detections.append(detection)
        
        if not target_detections:
            return
        
        # Update existing target tracks with high-frequency detections
        for track_id in list(self.active_target_tracks):
            if track_id not in self.tracked_objects:
                self.active_target_tracks.discard(track_id)
                continue
                
            track = self.tracked_objects[track_id]
            track_bbox = track['bbox']
            
            best_match = None
            best_iou = 0
            
            # Find best matching detection
            for detection in target_detections:
                iou = self.calculate_iou(track_bbox, detection['bbox'])
                if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                    best_iou = iou
                    best_match = detection
            
            # Update track if good match found
            if best_match:
                track['bbox'] = best_match['bbox']
                track['yolo_class'] = best_match['class_name']
                track['yolo_confidence'] = best_match['confidence']
                track['hits'] += 1
                track['age'] = 0
                track['last_seen'] = frame_count
                track['high_freq_updated'] = True  # Mark as high-freq updated
    
    def get_tracks_needing_classification(self):
        """Get tracks that need classification"""
        tracks_to_classify = []
        
        for track_id, track in self.tracked_objects.items():
            # Only classify if:
            # 1. Not already classified
            # 2. Has enough hits (stable track)
            # 3. Not too old
            if (not track['classified'] and 
                track['hits'] >= self.min_track_hits and 
                track['age'] < self.max_track_age):
                
                tracks_to_classify.append((track_id, track))
        
        return tracks_to_classify
    
    def get_processed_results(self):
        """Get all processed classification results from queue"""
        results = []
        
        try:
            while not self.results_queue.empty():
                result = self.results_queue.get_nowait()
                
                # Check if it's a target object
                is_target, target_type = self.is_target_object(result['mobilenet_class'])
                result['is_target'] = is_target
                result['target_type'] = target_type
                
                # Update the tracked object with classification results
                track_id = result['track_id']
                if track_id in self.tracked_objects:
                    self.tracked_objects[track_id]['classified'] = True
                    self.tracked_objects[track_id]['classification_result'] = result
                    self.tracked_objects[track_id]['is_target'] = is_target
                    self.tracked_objects[track_id]['target_type'] = target_type
                    
                    # Add to active target tracks for high-frequency updates
                    if is_target:
                        self.active_target_tracks.add(track_id)
                
                if is_target:
                    self.target_confirmations += 1
                    self.logger.info(f"TARGET CONFIRMED: {target_type} - {result['mobilenet_class']} (Track {track_id})")
                
                results.append(result)
                
        except Empty:
            pass
        
        return results
    
    def draw_tracked_objects(self, frame, tracked_objects):
        """Draw tracked objects on frame with improved label positioning"""
        height, width = frame.shape[:2]
        used_label_positions = []  # Track used label positions to prevent overlaps
        
        for track_id, track in tracked_objects.items():
            x1, y1, x2, y2 = track['bbox']
            
            # Determine if target object
            if track.get('is_target', False):
                color = (0, 255, 0)  # Green for targets
                thickness = 3
                label = f"TARGET: {track['target_type'].upper()}"
            elif track.get('classified', False):
                color = (0, 0, 255)  # Red for classified non-targets
                thickness = 2
                label = f"NON-TARGET: {track['yolo_class']}"
            else:
                color = (0, 165, 255)  # Orange for pending classification
                thickness = 2
                label = f"TRACKING: {track['yolo_class']}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Find a good position for the main label (avoid overlaps)
            label_y = self._find_best_label_position(y2, height, used_label_positions)
            used_label_positions.append(label_y)
            
            # Draw main label with background for better visibility
            self._draw_text_with_background(frame, label, (x1, label_y), color, (255, 255, 255))
            
            # Draw compact info above the box (only if there's space)
            if y1 > 60:
                # Compact track info
                track_info = f"ID:{track_id} H:{track['hits']}"
                cv2.putText(frame, track_info, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Compact YOLO info
                yolo_text = f"{track['yolo_class']} ({track['yolo_confidence']:.2f})"
                cv2.putText(frame, yolo_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Draw MobileNet info if available (compact)
            if track.get('classified', False) and track.get('classification_result'):
                classification = track['classification_result']
                mobile_text = f"{classification['mobilenet_class']} ({classification['mobilenet_confidence']:.2f})"
                if y1 > 40:
                    cv2.putText(frame, mobile_text, (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (139, 0, 0), 1)
    
    def _find_best_label_position(self, y2, frame_height, used_positions):
        """Find the best Y position for a label to avoid overlaps"""
        base_y = y2 + 20
        
        # If base position is good and not used, use it
        if base_y < frame_height - 30 and base_y not in used_positions:
            return base_y
        
        # Otherwise, find the next available position
        test_y = base_y
        while test_y < frame_height - 30:
            if test_y not in used_positions:
                return test_y
            test_y += 25
        
        # If no good position found, use base position
        return min(base_y, frame_height - 30)
    
    def _draw_text_with_background(self, frame, text, position, text_color, bg_color):
        """Draw text with a background rectangle for better visibility"""
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (x - 2, y - text_height - 2), 
                     (x + text_width + 2, y + baseline + 2), 
                     bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)
    
    def run_detection(self):
        """Main detection loop with async processing"""
        print("Opening camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            print("Failed to open camera!")
            return
        
        # Camera settings - optimized for maximum FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # Lower width for better FPS
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)  # Lower height for better FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        cap.set(cv2.CAP_PROP_FPS, 60)  # Higher target frame rate
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {width}x{height}")
        
        # Start classification worker thread
        self.running = True
        self.classifier_thread = Thread(target=self.classification_worker, daemon=True)
        self.classifier_thread.start()
        
        print("Tracked Adaptive Async detection pipeline running! Press 'q' to quit")
        print("=" * 60)
        
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        # Smooth FPS calculation using rolling average
        fps_history = deque(maxlen=30)  # Keep last 30 FPS measurements
        current_fps = 0
        
        # Detection timing control - now adaptive
        last_detection_time = 0
        
        # Tracking statistics
        total_tracks_created = 0
        total_tracks_classified = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Detect motion and update detection interval
                motion_detected, motion_level = self.detect_motion(frame)
                self.update_detection_interval(motion_detected)
                
                # Run YOLO detection at adaptive interval
                if current_time - last_detection_time >= self.current_detection_interval:
                    # Enhance frame for small object detection
                    enhanced_frame = self.enhance_image_for_small_objects(frame)
                    
                    # Stage 1: YOLO Detection
                    detections = self.detect_objects_yolo(enhanced_frame)
                    
                    # Update tracks with new detections
                    active_track_ids = self.update_tracks(detections, frame_count)
                    
                    # Get tracks that need classification
                    tracks_to_classify = self.get_tracks_needing_classification()
                    
                    # Queue tracks for classification
                    for track_id, track_info in tracks_to_classify:
                        success = self.queue_for_classification(
                            frame, track_id, track_info, current_time
                        )
                        if success:
                            total_tracks_created += 1
                    
                    self.total_detections += len(detections)
                    last_detection_time = current_time
                
                # High-frequency target object detection (10Hz) - simplified approach
                if current_time - self.last_target_detection_time >= self.target_detection_interval:
                    # Quick detection for target objects using existing method
                    if self.active_target_tracks:
                        # Use existing detection method with lower confidence for speed
                        quick_detections = self.detect_objects_yolo(frame)  # No enhancement for speed
                        
                        # Update confirmed target tracks with high-frequency detection
                        self.update_target_tracks_high_freq(quick_detections, frame_count)
                    
                    self.last_target_detection_time = current_time
                
                # Get processed classification results
                processed_results = self.get_processed_results()
                
                # Count newly classified tracks
                for result in processed_results:
                    if result['track_id'] in self.tracked_objects:
                        if self.tracked_objects[result['track_id']].get('classified', False):
                            total_tracks_classified += 1
                        
                        if result['is_target']:
                            self.target_objects_found += 1
                
                # Draw tracked objects
                self.draw_tracked_objects(frame, self.tracked_objects)
                
                # Calculate and display FPS with smooth rolling average
                fps_counter += 1
                current_time_for_fps = time.time()
                
                if current_time_for_fps - fps_start_time >= 1.0:
                    # Calculate current FPS
                    instant_fps = fps_counter
                    fps_history.append(instant_fps)
                    
                    # Calculate rolling average FPS
                    current_fps = sum(fps_history) / len(fps_history)
                    
                    # Reset counters
                    fps_counter = 0
                    fps_start_time = current_time_for_fps
                
                # Display stats with smooth FPS and motion info
                queue_size = self.classification_queue.qsize()
                motion_status = "MOTION" if motion_detected else "STATIC"
                detection_freq = f"{1/self.current_detection_interval:.1f}Hz"
                active_tracks = len(self.tracked_objects)
                stats_text = f"FPS: {current_fps:.1f} | {motion_status} | {detection_freq} | Tracks: {active_tracks}"
                cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Additional stats line
                active_targets = len(self.active_target_tracks)
                stats_text2 = f"Detections: {self.total_detections} | Targets: {self.target_objects_found} | Active: {active_targets} | Classified: {total_tracks_classified}"
                cv2.putText(frame, stats_text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Pipeline info
                model_display_name = "YOLOv11s" if "yolo11s" in self.selected_model else "YOLOv8n"
                cv2.putText(frame, f"HIGH FPS PIPELINE: {model_display_name} → Streamlined Tracking → MobileNet", 
                           (10, height-120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "GREEN: Target Objects | RED: Non-Targets | ORANGE: Tracking", 
                           (10, height-100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Motion Level: {motion_level:.0f} | Threshold: {self.motion_threshold}", 
                           (10, height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Small Objects: 2 scales | Min Size: {self.min_box_size}px | Max Detections: {self.max_detections}", 
                           (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Confidence: {self.yolo_confidence:.2f} | IoU: 0.5 | Fast Tracking: ON", 
                           (10, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Target Detection: 10Hz (100ms) | Active Targets: {active_targets}", 
                           (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Tracks Created: {total_tracks_created} | Confirmations: {self.target_confirmations}", 
                           (10, height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Async Object Detection Pipeline', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Cleanup
            self.running = False
            if self.classifier_thread:
                self.classifier_thread.join(timeout=2.0)
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            print("\n" + "=" * 60)
            print("HIGH FPS OPTIMIZED DETECTION PIPELINE SUMMARY")
            print("=" * 60)
            print(f"YOLO Model used: {self.selected_model}")
            print(f"Total detections: {self.total_detections}")
            print(f"Target objects found: {self.target_objects_found}")
            print(f"Classifications performed: {self.classifications_performed}")
            print(f"Target confirmations: {self.target_confirmations}")
            print(f"Tracks created: {total_tracks_created}")
            print(f"Tracks classified: {total_tracks_classified}")
            print(f"Max queue size reached: {self.queue_stats['max_queue_size']}")
            print(f"Total queued: {self.queue_stats['total_queued']}")
            print(f"Total processed: {self.queue_stats['total_processed']}")
            print(f"Active tracks at end: {len(self.tracked_objects)}")
            
            if self.total_detections > 0:
                target_rate = (self.target_objects_found / self.total_detections) * 100
                print(f"Target detection rate: {target_rate:.1f}%")
            
            print("=" * 60)

def main():
    """Main function"""
    try:
        # You can adjust these parameters:
        # - max_queue_size: Maximum number of items in classification queue
        # - batch_size: Number of images to classify together (GPU batch processing)
        detector = AsyncObjectDetectionPipeline(max_queue_size=30, batch_size=4)
        detector.run_detection()
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
