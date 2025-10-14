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

class AsyncObjectDetectionPipeline:
    def __init__(self, max_queue_size=30, batch_size=4):
        print("=" * 60)
        print("ASYNC OBJECT DETECTION & CLASSIFICATION PIPELINE")
        print("=" * 60)
        print("Architecture: Queue-based Async Processing")
        print("Stage 1: YOLOv8 Object Detection (GPU)")
        print("Stage 2: MobileNetV3 Classification (GPU)")
        print("Processing: Async Queue with Batch Processing")
        print("=" * 60)
        
        self.setup_logging()
        
        self.setup_gpu()
        
        print("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')
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
        
        # Detection settings
        self.yolo_confidence = 0.4
        self.mobilenet_confidence = 0.3
        self.max_detections = 10
        
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
        self.logger.info(f"YOLO Device: {self.device.upper()}")
        self.logger.info(f"TensorFlow/MobileNet Device: GPU Enabled")
        self.logger.info(f"Queue max size: {max_queue_size}, Batch size: {batch_size}")
        self.logger.info(f"Target objects: {list(self.target_objects.keys())}")
    
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
        """Stage 1: Use YOLOv8 for object detection"""
        try:
            results = self.yolo_model(
                frame,
                conf=self.yolo_confidence,
                iou=0.5,
                max_det=self.max_detections,
                verbose=False,
                device=self.device
            )
            
            if not results or len(results) == 0:
                return []
            
            detections = []
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                raw_detections = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                    confidence = float(box.conf[0].cpu())
                    class_id = int(box.cls[0].cpu())
                    class_name = self.yolo_model.names[class_id]
                    
                    # # Skip very small detections
                    # if (x2 - x1) < 30 or (y2 - y1) < 30:
                    #     continue
                    
                    raw_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_name': class_name,
                        'class_id': class_id
                    })
                
                # Deduplicate overlapping detections
                detections = self.deduplicate_detections(raw_detections)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
            return []
    
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
    
    def queue_for_classification(self, frame, detection, detection_id, timestamp):
        """Queue a detected object for MobileNet classification"""
        try:
            x1, y1, x2, y2 = detection['bbox']
            cropped_obj = frame[y1:y2, x1:x2]
            
            if cropped_obj.size == 0:
                return False
            
            # Create classification task
            task = {
                'detection_id': detection_id,
                'cropped_image': cropped_obj.copy(),
                'bbox': detection['bbox'],
                'yolo_class': detection['class_name'],
                'yolo_confidence': detection['confidence'],
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
                        'detection_id': task['detection_id'],
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
        batch_timeout = 0.05  # 50ms timeout to process partial batches
        
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
                
                if is_target:
                    self.target_confirmations += 1
                    self.logger.info(f"TARGET CONFIRMED: {target_type} - {result['mobilenet_class']}")
                
                results.append(result)
                
        except Empty:
            pass
        
        return results
    
    def draw_detections(self, frame, detections):
        """Draw detection results on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Determine if target object
            if detection.get('is_target', False):
                color = (0, 255, 0)  # Green for targets
                thickness = 3
                label = f"TARGET: {detection['target_type'].upper()}"
            else:
                color = (0, 165, 255)  # Orange for pending classification
                thickness = 2
                label = f"YOLO: {detection['yolo_class']}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw YOLO info
            yolo_text = f"YOLO: {detection['yolo_class']} ({detection['yolo_confidence']:.2f})"
            cv2.putText(frame, yolo_text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw MobileNet info if available
            if 'mobilenet_class' in detection:
                mobile_text = f"MobileNet: {detection['mobilenet_class']} ({detection['mobilenet_confidence']:.2f})"
                cv2.putText(frame, mobile_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 0, 0), 1)
                
                # Draw final label
                cv2.putText(frame, label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Show "Processing..." for queued items
                cv2.putText(frame, "Processing...", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    
    def run_detection(self):
        """Main detection loop with async processing"""
        print("Opening camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            print("Failed to open camera!")
            return
        
        # Camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {width}x{height}")
        
        # Start classification worker thread
        self.running = True
        self.classifier_thread = Thread(target=self.classification_worker, daemon=True)
        self.classifier_thread.start()
        
        print("Async detection pipeline running! Press 'q' to quit")
        print("=" * 60)
        
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        # Smooth FPS calculation using rolling average
        fps_history = deque(maxlen=30)  # Keep last 30 FPS measurements
        current_fps = 0
        
        # Detection timing control
        last_detection_time = 0
        detection_interval = 0.1  # 100ms = 10 times per second
        
        # Active detections tracking
        active_detections = {}  # detection_id -> detection_info
        detection_ttl = 2.0  # Keep detections for 2 seconds max
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Clean up old detections (expired after TTL)
                expired_ids = [
                    det_id for det_id, det in active_detections.items()
                    if current_time - det['timestamp'] > detection_ttl
                ]
                for det_id in expired_ids:
                    del active_detections[det_id]
                
                # Run YOLO detection at controlled interval
                if current_time - last_detection_time >= detection_interval:
                    # Stage 1: YOLO Detection
                    detections = self.detect_objects_yolo(frame)
                    
                    # Add new detections (don't clear old ones that are still being processed)
                    for detection in detections:
                        detection_id = str(uuid.uuid4())
                        
                        # Store detection info
                        active_detections[detection_id] = {
                            'bbox': detection['bbox'],
                            'yolo_class': detection['class_name'],
                            'yolo_confidence': detection['confidence'],
                            'timestamp': current_time,
                            'classified': False  # Track if classification is complete
                        }
                        
                        # Queue for classification
                        self.queue_for_classification(
                            frame, detection, detection_id, current_time
                        )
                        
                        self.total_detections += 1
                    
                    last_detection_time = current_time
                
                # Get processed classification results
                processed_results = self.get_processed_results()
                
                # Update active detections with classification results
                for result in processed_results:
                    detection_id = result['detection_id']
                    if detection_id in active_detections:
                        active_detections[detection_id].update({
                            'mobilenet_class': result['mobilenet_class'],
                            'mobilenet_confidence': result['mobilenet_confidence'],
                            'is_target': result['is_target'],
                            'target_type': result['target_type'],
                            'queue_time': result['queue_time'],
                            'classified': True  # Mark as classified
                        })
                        
                        if result['is_target']:
                            self.target_objects_found += 1
                
                # Draw only the most recent classified detection for each bbox area
                # This prevents overlapping "Processing..." and classification text
                detections_to_draw = []
                drawn_areas = []
                
                # Sort by timestamp (newest first) and classification status (classified first)
                sorted_detections = sorted(
                    active_detections.items(),
                    key=lambda x: (x[1].get('classified', False), x[1]['timestamp']),
                    reverse=True
                )
                
                for det_id, det in sorted_detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Check if this area already has a detection drawn
                    overlaps = False
                    for drawn_bbox in drawn_areas:
                        dx1, dy1, dx2, dy2 = drawn_bbox
                        # Calculate overlap
                        overlap_x1 = max(x1, dx1)
                        overlap_y1 = max(y1, dy1)
                        overlap_x2 = min(x2, dx2)
                        overlap_y2 = min(y2, dy2)
                        
                        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                            bbox_area = (x2 - x1) * (y2 - y1)
                            if overlap_area / bbox_area > 0.5:  # 50% overlap
                                overlaps = True
                                break
                    
                    if not overlaps:
                        detections_to_draw.append(det)
                        drawn_areas.append(bbox)
                
                self.draw_detections(frame, detections_to_draw)
                
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
                
                # Display stats with smooth FPS
                queue_size = self.classification_queue.qsize()
                stats_text = f"FPS: {current_fps:.1f} | Queue: {queue_size} | Detections: {self.total_detections} | Targets: {self.target_objects_found}"
                cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Pipeline info
                cv2.putText(frame, "ASYNC PIPELINE: YOLO (GPU) → Queue → MobileNet Batch (GPU)", 
                           (10, height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "GREEN: Target Objects | ORANGE: Processing", 
                           (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Batch Size: {self.batch_size} | Max Queue: {self.queue_stats['max_queue_size']}", 
                           (10, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Processed: {self.queue_stats['total_processed']} | Confirmations: {self.target_confirmations}", 
                           (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
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
            print("ASYNC PIPELINE SUMMARY")
            print("=" * 60)
            print(f"Total detections: {self.total_detections}")
            print(f"Target objects found: {self.target_objects_found}")
            print(f"Classifications performed: {self.classifications_performed}")
            print(f"Target confirmations: {self.target_confirmations}")
            print(f"Max queue size reached: {self.queue_stats['max_queue_size']}")
            print(f"Total queued: {self.queue_stats['total_queued']}")
            print(f"Total processed: {self.queue_stats['total_processed']}")
            
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
