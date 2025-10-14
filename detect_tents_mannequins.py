#!/usr/bin/env python3

"""
Fine-Tuned Object Detection Pipeline for Tents and Mannequins
Optimized for maximum accuracy with async GPU processing.

Key Optimizations:
- YOLO pre-filtering: Only process person/tent-related classes
- Lower confidence thresholds for target classes
- MobileNet focused classification with expanded keyword matching
- Multi-stage verification for higher accuracy
- Batch processing for GPU efficiency
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
import uuid

class TentMannequinDetector:
    def __init__(self, max_queue_size=30, batch_size=4):
        print("=" * 70)
        print("FINE-TUNED TENT & MANNEQUIN DETECTION PIPELINE")
        print("=" * 70)
        print("🎯 Target Classes: Tents, Mannequins")
        print("🚀 Architecture: Async GPU Processing with Smart Filtering")
        print("=" * 70)
        
        self.setup_logging()
        self.setup_gpu()
        
        # Load YOLO model
        print("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')
        if self.device == 'mps':
            self.yolo_model.to(self.device)
        
        # Load MobileNet model
        print("Loading MobileNetV3 model...")
        self.setup_tensorflow_gpu()
        self.mobilenet_model = MobileNetV3Small(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=True
        )
        
        # Queue system
        self.classification_queue = Queue(maxsize=max_queue_size)
        self.results_queue = Queue(maxsize=max_queue_size * 2)
        self.batch_size = batch_size
        
        # Processing control
        self.running = False
        self.classifier_thread = None
        
        # YOLO classes that might be tents or mannequins
        # Pre-filter to only process these classes
        self.yolo_target_classes = {
            'person',      # Could be mannequin
            'backpack',    # Often near tents
            'suitcase',    # Display mannequins
            'umbrella',    # Tent-like shapes
            'handbag',     # Mannequin accessories
            'tie',         # Mannequin clothing
            'sports ball', # Camping gear
            'kite',        # Tent-like shapes
            'bed',         # Tent interior
            'couch',       # Display furniture
            'chair',       # Display furniture
            'potted plant',# Display setup
            'vase',        # Display items
            'bench'        # Outdoor furniture near tents
        }
        
        # Expanded target keywords for better matching
        self.target_keywords = {
            'tent': [
                # Common tent types
                'tent', 'camping_tent', 'canopy', 'pavilion', 'marquee',
                'yurt', 'teepee', 'tipi', 'wigwam',
                # Tent-related
                'shelter', 'dome', 'canvas', 'tarp', 'awning',
                # Mountain/camping tents
                'mountain_tent', 'backpacking_tent', 'camping',
                # Event tents
                'party_tent', 'wedding_tent', 'event_tent',
                # Specific styles
                'pup_tent', 'ridge_tent', 'tunnel_tent', 'dome_tent',
                'bell_tent', 'safari_tent', 'geodesic_tent'
            ],
            'mannequin': [
                # Direct matches
                'mannequin', 'manikin', 'mannikin', 'dummy',
                # Display-related
                'display_model', 'dress_form', 'tailor_dummy',
                'store_dummy', 'shop_dummy', 'retail_dummy',
                # Fashion-related
                'fashion_model', 'clothing_model', 'garment_form',
                # Types
                'male_mannequin', 'female_mannequin', 'child_mannequin',
                'torso_mannequin', 'full_body_mannequin',
                # Related items
                'dressmaker', 'bust', 'figure', 'sculpture',
                'statue', 'effigy', 'form'
            ]
        }
        
        # Fine-tuned confidence thresholds
        self.yolo_confidence_target = 0.25   # Lower for potential tent/mannequin
        self.yolo_confidence_general = 0.5   # Higher for other objects
        self.mobilenet_confidence_threshold = 0.15  # Lower threshold for better recall
        self.min_detection_size = 40  # Minimum bbox size
        self.max_detections = 10  # Maximum detections per frame
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'yolo_detections': 0,
            'filtered_detections': 0,
            'classifications_performed': 0,
            'tents_found': 0,
            'mannequins_found': 0,
            'false_positives': 0,
            'queue_max': 0
        }
        
        self.logger.info("✓ Models loaded successfully!")
        self.logger.info(f"✓ YOLO Device: {self.device.upper()}")
        self.logger.info(f"✓ Pre-filter classes: {len(self.yolo_target_classes)}")
        self.logger.info(f"✓ Tent keywords: {len(self.target_keywords['tent'])}")
        self.logger.info(f"✓ Mannequin keywords: {len(self.target_keywords['mannequin'])}")
    
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
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ TensorFlow GPU enabled: {len(gpus)} GPU(s)")
                self.logger.info(f"TensorFlow GPUs: {gpus}")
            except RuntimeError as e:
                print(f"⚠ TensorFlow GPU setup warning: {e}")
        else:
            print(f"ℹ TensorFlow using optimized CPU/Metal")
    
    def setup_logging(self):
        """Setup logging system"""
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger = logging.getLogger('TentMannequinDetector')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(f'logs/tent_mannequin_{timestamp}.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        print(f"Logging to: logs/tent_mannequin_{timestamp}.log")
    
    def is_target_class(self, class_name):
        """Check if YOLO class is worth processing"""
        return class_name.lower() in self.yolo_target_classes
    
    def detect_objects_yolo(self, frame):
        """Stage 1: YOLO detection with smart pre-filtering"""
        try:
            # Run YOLO with different confidence for different classes
            results = self.yolo_model(
                frame,
                conf=self.yolo_confidence_target,  # Lower threshold
                iou=0.4,
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
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                    confidence = float(box.conf[0].cpu())
                    class_id = int(box.cls[0].cpu())
                    class_name = self.yolo_model.names[class_id]
                    
                    # Size filter
                    width = x2 - x1
                    height = y2 - y1
                    if width < self.min_detection_size or height < self.min_detection_size:
                        continue
                    
                    # Smart filtering: Only process target classes
                    if self.is_target_class(class_name):
                        # For target-relevant classes, accept lower confidence
                        if confidence >= self.yolo_confidence_target:
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_name': class_name,
                                'class_id': class_id
                            })
                            self.logger.info(f"YOLO: {class_name} ({confidence:.3f}) - QUEUED FOR CLASSIFICATION")
                    else:
                        # For non-target classes, require high confidence
                        if confidence >= self.yolo_confidence_general:
                            # Still process but mark as unlikely
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_name': class_name,
                                'class_id': class_id
                            })
                
                self.stats['yolo_detections'] += len(detections)
                
                # Deduplicate
                detections = self.deduplicate_detections(detections)
                self.stats['filtered_detections'] += len(detections)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
            return []
    
    def deduplicate_detections(self, raw_detections):
        """Remove overlapping detections"""
        if not raw_detections:
            return []
        
        # Sort by confidence
        sorted_detections = sorted(raw_detections, key=lambda x: x['confidence'], reverse=True)
        filtered = []
        
        for detection in sorted_detections:
            x1, y1, x2, y2 = detection['bbox']
            is_duplicate = False
            
            for accepted in filtered:
                ax1, ay1, ax2, ay2 = accepted['bbox']
                
                # Calculate IoU
                ix1, iy1 = max(x1, ax1), max(y1, ay1)
                ix2, iy2 = min(x2, ax2), min(y2, ay2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (ax2 - ax1) * (ay2 - ay1)
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.3:
                        is_duplicate = True
                        break
            
            if not is_duplicate and len(filtered) < 5:  # Max 5 detections
                filtered.append(detection)
        
        return filtered
    
    def queue_for_classification(self, frame, detection, detection_id, timestamp):
        """Queue detected object for MobileNet classification"""
        try:
            x1, y1, x2, y2 = detection['bbox']
            
            # Add padding for better classification
            padding = 10
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return False
            
            task = {
                'detection_id': detection_id,
                'cropped_image': cropped.copy(),
                'bbox': detection['bbox'],
                'yolo_class': detection['class_name'],
                'yolo_confidence': detection['confidence'],
                'timestamp': timestamp,
                'queued_at': time.time()
            }
            
            if not self.classification_queue.full():
                self.classification_queue.put(task, block=False)
                current_size = self.classification_queue.qsize()
                if current_size > self.stats['queue_max']:
                    self.stats['queue_max'] = current_size
                return True
            else:
                self.logger.warning("Queue full, skipping detection")
                return False
                
        except Exception as e:
            self.logger.error(f"Queue error: {e}")
            return False
    
    def classify_batch(self, batch_tasks):
        """Batch classification with MobileNet"""
        if not batch_tasks:
            return []
        
        try:
            batch_images = []
            for task in batch_tasks:
                # Preprocess
                resized = cv2.resize(task['cropped_image'], (224, 224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(rgb)
                img_array = image.img_to_array(pil_img)
                img_array = preprocess_input(img_array)
                batch_images.append(img_array)
            
            # Batch prediction
            batch_array = np.array(batch_images)
            predictions = self.mobilenet_model.predict(batch_array, verbose=0)
            
            # Process results
            results = []
            for i, task in enumerate(batch_tasks):
                pred = predictions[i:i+1]
                decoded = decode_predictions(pred, top=5)[0]  # Top 5 for better matching
                
                self.stats['classifications_performed'] += 1
                
                # Check all top predictions for target keywords
                is_target = False
                target_type = None
                best_match = None
                best_confidence = 0.0
                
                for class_id, class_name, confidence in decoded:
                    class_lower = class_name.lower()
                    
                    # Check tent keywords
                    for keyword in self.target_keywords['tent']:
                        if keyword.lower() in class_lower or class_lower in keyword.lower():
                            if confidence > best_confidence:
                                is_target = True
                                target_type = 'tent'
                                best_match = class_name
                                best_confidence = confidence
                    
                    # Check mannequin keywords
                    for keyword in self.target_keywords['mannequin']:
                        if keyword.lower() in class_lower or class_lower in keyword.lower():
                            if confidence > best_confidence:
                                is_target = True
                                target_type = 'mannequin'
                                best_match = class_name
                                best_confidence = confidence
                
                # Apply confidence threshold
                if is_target and best_confidence < self.mobilenet_confidence_threshold:
                    is_target = False
                    target_type = None
                
                result = {
                    'detection_id': task['detection_id'],
                    'bbox': task['bbox'],
                    'yolo_class': task['yolo_class'],
                    'yolo_confidence': task['yolo_confidence'],
                    'mobilenet_class': decoded[0][1],
                    'mobilenet_confidence': decoded[0][2],
                    'all_predictions': decoded,
                    'is_target': is_target,
                    'target_type': target_type,
                    'matched_class': best_match if is_target else None,
                    'match_confidence': best_confidence if is_target else 0.0,
                    'queue_time': time.time() - task['queued_at']
                }
                
                if is_target:
                    if target_type == 'tent':
                        self.stats['tents_found'] += 1
                    elif target_type == 'mannequin':
                        self.stats['mannequins_found'] += 1
                    
                    self.logger.info(f"🎯 TARGET FOUND: {target_type.upper()} - {best_match} ({best_confidence:.3f})")
                else:
                    self.stats['false_positives'] += 1
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch classification error: {e}")
            return []
    
    def classification_worker(self):
        """Background worker for async classification"""
        self.logger.info("Classification worker started")
        batch_buffer = []
        last_process_time = time.time()
        batch_timeout = 0.05
        
        while self.running:
            try:
                try:
                    task = self.classification_queue.get(timeout=0.01)
                    batch_buffer.append(task)
                except Empty:
                    pass
                
                current_time = time.time()
                time_since_last = current_time - last_process_time
                
                should_process = (
                    len(batch_buffer) >= self.batch_size or
                    (len(batch_buffer) > 0 and time_since_last >= batch_timeout) or
                    (len(batch_buffer) > 0 and self.classification_queue.empty())
                )
                
                if should_process and batch_buffer:
                    results = self.classify_batch(batch_buffer)
                    
                    for result in results:
                        self.results_queue.put(result)
                    
                    self.logger.info(f"Processed batch of {len(batch_buffer)} objects")
                    batch_buffer.clear()
                    last_process_time = current_time
                
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                batch_buffer.clear()
        
        self.logger.info("Classification worker stopped")
    
    def get_processed_results(self):
        """Get processed classification results"""
        results = []
        
        try:
            while not self.results_queue.empty():
                result = self.results_queue.get_nowait()
                results.append(result)
        except Empty:
            pass
        
        return results
    
    def draw_detections(self, frame, detections):
        """Draw detection results with enhanced visualization"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            if det.get('is_target', False):
                # Target found - bright green
                color = (0, 255, 0)
                thickness = 3
                target_type = det['target_type'].upper()
                
                # Draw banner background
                cv2.rectangle(frame, (x1, y1-35), (x2, y1), (0, 200, 0), -1)
                
                label = f"✓ {target_type}"
                match_class = det.get('matched_class', det['mobilenet_class'])
                conf = det.get('match_confidence', det['mobilenet_confidence'])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, label, (x1+5, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"{match_class} ({conf:.2f})", (x1+5, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
            elif 'mobilenet_class' in det:
                # Classified but not target - red
                color = (0, 0, 255)
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"X Not target", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Processing - orange
                color = (0, 165, 255)
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, "Processing...", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def run_detection(self):
        """Main detection loop"""
        print("Opening camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            print("Failed to open camera!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: {width}x{height}")
        
        # Start worker
        self.running = True
        self.classifier_thread = Thread(target=self.classification_worker, daemon=True)
        self.classifier_thread.start()
        
        print("🎯 Detection active! Press 'q' to quit")
        print("=" * 70)
        
        frame_count = 0
        fps_counter = 0
        fps_start = time.time()
        
        last_detection_time = 0
        detection_interval = 0.1
        
        active_detections = {}
        detection_ttl = 2.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                self.stats['total_frames'] = frame_count
                current_time = time.time()
                
                # Clean expired detections
                expired = [
                    det_id for det_id, det in active_detections.items()
                    if current_time - det['timestamp'] > detection_ttl
                ]
                for det_id in expired:
                    del active_detections[det_id]
                
                # Run YOLO detection
                if current_time - last_detection_time >= detection_interval:
                    detections = self.detect_objects_yolo(frame)
                    
                    for detection in detections:
                        detection_id = str(uuid.uuid4())
                        
                        active_detections[detection_id] = {
                            'bbox': detection['bbox'],
                            'yolo_class': detection['class_name'],
                            'yolo_confidence': detection['confidence'],
                            'timestamp': current_time,
                            'classified': False
                        }
                        
                        self.queue_for_classification(frame, detection, detection_id, current_time)
                    
                    last_detection_time = current_time
                
                # Get classification results
                processed = self.get_processed_results()
                
                for result in processed:
                    det_id = result['detection_id']
                    if det_id in active_detections:
                        active_detections[det_id].update({
                            'mobilenet_class': result['mobilenet_class'],
                            'mobilenet_confidence': result['mobilenet_confidence'],
                            'is_target': result['is_target'],
                            'target_type': result['target_type'],
                            'matched_class': result.get('matched_class'),
                            'match_confidence': result.get('match_confidence', 0.0),
                            'classified': True
                        })
                
                # Draw only most recent per area
                to_draw = []
                drawn_areas = []
                
                sorted_dets = sorted(
                    active_detections.items(),
                    key=lambda x: (x[1].get('classified', False), x[1]['timestamp']),
                    reverse=True
                )
                
                for det_id, det in sorted_dets:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    overlaps = False
                    for drawn_bbox in drawn_areas:
                        dx1, dy1, dx2, dy2 = drawn_bbox
                        ox1, oy1 = max(x1, dx1), max(y1, dy1)
                        ox2, oy2 = min(x2, dx2), min(y2, dy2)
                        
                        if ox1 < ox2 and oy1 < oy2:
                            overlap_area = (ox2 - ox1) * (oy2 - oy1)
                            bbox_area = (x2 - x1) * (y2 - y1)
                            if overlap_area / bbox_area > 0.5:
                                overlaps = True
                                break
                    
                    if not overlaps:
                        to_draw.append(det)
                        drawn_areas.append(bbox)
                
                self.draw_detections(frame, to_draw)
                
                # FPS
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_start = time.time()
                    
                    queue_size = self.classification_queue.qsize()
                    stats_text = f"FPS: {fps} | Queue: {queue_size} | Tents: {self.stats['tents_found']} | Mannequins: {self.stats['mannequins_found']}"
                    cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Info overlay
                cv2.putText(frame, "🎯 TENT & MANNEQUIN DETECTOR | Optimized for Accuracy", 
                           (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(frame, "GREEN: Target Found | RED: Not Target | ORANGE: Processing", 
                           (10, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(frame, f"Processed: {self.stats['classifications_performed']} | Accuracy Optimized", 
                           (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
                cv2.imshow('Tent & Mannequin Detector', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.running = False
            if self.classifier_thread:
                self.classifier_thread.join(timeout=2.0)
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Final stats
            print("\n" + "=" * 70)
            print("DETECTION SUMMARY")
            print("=" * 70)
            print(f"Total frames: {self.stats['total_frames']}")
            print(f"YOLO detections: {self.stats['yolo_detections']}")
            print(f"Filtered detections: {self.stats['filtered_detections']}")
            print(f"Classifications: {self.stats['classifications_performed']}")
            print(f"🎪 Tents found: {self.stats['tents_found']}")
            print(f"👤 Mannequins found: {self.stats['mannequins_found']}")
            print(f"False positives: {self.stats['false_positives']}")
            print(f"Max queue size: {self.stats['queue_max']}")
            
            if self.stats['classifications_performed'] > 0:
                accuracy = ((self.stats['tents_found'] + self.stats['mannequins_found']) / 
                           self.stats['classifications_performed']) * 100
                print(f"Target detection rate: {accuracy:.1f}%")
            
            print("=" * 70)

def main():
    """Main function"""
    try:
        # Adjust batch_size based on GPU memory:
        # - Jetson Nano: batch_size=2
        # - Mac M1/M2: batch_size=4
        # - High-end GPU: batch_size=8
        detector = TentMannequinDetector(max_queue_size=30, batch_size=4)
        detector.run_detection()
    except KeyboardInterrupt:
        print("\n⚠ Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
