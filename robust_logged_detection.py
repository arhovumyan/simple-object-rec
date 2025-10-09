#!/usr/bin/env python3

"""
Robust Logged Combined YOLO + MobileNetV3 Detection
Fixed error handling and better MobileNet classification
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
import logging
import json
from datetime import datetime
import os

class RobustLoggedDetector:
    def __init__(self):
        print("=" * 60)
        print("ROBUST Logged Combined YOLO + MobileNetV3 Detection")
        print("=" * 60)
        
        # Setup logging
        self.setup_logging()
        
        # Setup devices
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.logger.info(f"YOLO Device: {self.device.upper()}")
        
        # Load models
        self.logger.info("Loading models...")
        self.yolo_model = YOLO('yolov8n.pt')
        if self.device == 'mps':
            self.yolo_model.to(self.device)
        
        tf.config.set_visible_devices([], 'GPU')
        self.mobilenet_model = MobileNetV3Small(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=True
        )
        self.logger.info("Models loaded successfully!")
        
        # Target object mappings (MobileNet-friendly objects)
        self.target_objects = {
            'phone': ['cellular_telephone', 'cellular_phone', 'cellphone', 'mobile_phone', 'telephone'],
            'mouse': ['computer_mouse', 'mouse', 'trackball'],
            'laptop': ['laptop', 'notebook', 'portable_computer'],
            'bottle': ['bottle', 'water_bottle', 'wine_bottle'],
            'cup': ['cup', 'coffee_cup', 'mug'],
            'book': ['book', 'notebook', 'magazine'],
            'keyboard': ['keyboard', 'computer_keyboard'],
            'chair': ['chair', 'office_chair', 'dining_chair'],
            'tv': ['television', 'tv', 'monitor', 'computer_screen']
            # Removed 'person' - MobileNetV3 is not good at person classification
        }
        
        # Performance settings
        self.yolo_conf = 0.4
        self.mobilenet_conf = 0.5  # Much higher threshold to filter bad classifications
        self.max_detections = 8
        
        # Temporal smoothing to reduce flickering
        self.classification_history = {}  # Store recent classifications per object
        self.coordinate_history = {}  # Store recent bounding box coordinates per object
        self.confidence_history = {}  # Store recent confidence values per object
        self.object_persistence = {}  # Store last seen frame for each object
        self.history_size = 8  # Keep last 8 classifications for stronger smoothing
        self.persistence_frames = 10  # Keep showing object for 10 frames after last detection
        
        # Statistics
        self.session_start = datetime.now()
        self.frame_count = 0
        self.total_detections = 0
        self.target_objects_found = 0
        self.mobilenet_verified = 0
        self.yolo_only_detections = 0
        
        # Log session start
        self.logger.info(f"Detection session started at {self.session_start}")
        self.logger.info(f"Target objects: {list(self.target_objects.keys())}")
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger = logging.getLogger('RobustDetector')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(f'logs/robust_detection_{timestamp}.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler (only warnings/errors)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # JSON logging
        self.json_log_file = f'logs/robust_detections_{timestamp}.jsonl'
        self.json_log = open(self.json_log_file, 'w')
        
        print(f"Logging to: logs/robust_detection_{timestamp}.log")
        print(f"JSON logs to: logs/robust_detections_{timestamp}.jsonl")
    
    def classify_with_mobilenet(self, cropped_image):
        """Robust MobileNetV3 classification"""
        try:
            # Resize to MobileNet input size
            resized_image = cv2.resize(cropped_image, (224, 224))
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL and preprocess
            pil_image = PILImage.fromarray(rgb_image)
            img_array = image.img_to_array(pil_image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Make prediction
            with tf.device('/CPU:0'):
                predictions = self.mobilenet_model.predict(img_array, verbose=0)
            
            # Decode predictions
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            
            # Log top 3 predictions (reduced for less spam)
            self.logger.info(f"MobileNet top 3 predictions:")
            for i, (class_id, class_name, confidence) in enumerate(decoded_predictions[:3]):
                self.logger.info(f"  {i+1}. {class_name}: {confidence:.3f}")
            
            # Return the top prediction regardless of threshold
            if decoded_predictions:
                top_prediction = decoded_predictions[0]
                return {
                    'class': top_prediction[1],
                    'confidence': top_prediction[2],
                    'all_predictions': decoded_predictions
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"MobileNet classification error: {e}")
            return None
    
    def is_target_object(self, class_name):
        """Check if object is one of our target objects"""
        class_lower = class_name.lower()
        for target, keywords in self.target_objects.items():
            if any(keyword in class_lower for keyword in keywords):
                return True, target
        return False, None
    
    def is_reasonable_classification(self, yolo_class, mobilenet_class):
        """Check if MobileNet classification makes sense for the YOLO detection"""
        yolo_lower = yolo_class.lower()
        mobilenet_lower = mobilenet_class.lower()
        
        # Person detection - MobileNet is terrible at this, reject most classifications
        if yolo_lower == 'person':
            # Only accept very specific, high-confidence classifications for people
            reasonable_person_classes = ['person', 'human', 'man', 'woman', 'child', 'baby']
            return any(reasonable in mobilenet_lower for reasonable in reasonable_person_classes)
        
        # Phone detection - should be reasonable
        if yolo_lower in ['cell phone', 'phone']:
            reasonable_phone_classes = ['phone', 'cellular', 'mobile', 'telephone', 'smartphone']
            return any(reasonable in mobilenet_lower for reasonable in reasonable_phone_classes)
        
        # Laptop detection - should be reasonable  
        if yolo_lower in ['laptop', 'computer']:
            reasonable_laptop_classes = ['laptop', 'computer', 'notebook', 'pc', 'macbook']
            return any(reasonable in mobilenet_lower for reasonable in reasonable_laptop_classes)
        
        # Bottle detection - should be reasonable
        if yolo_lower in ['bottle', 'wine glass']:
            reasonable_bottle_classes = ['bottle', 'wine', 'glass', 'cup', 'mug', 'drink']
            return any(reasonable in mobilenet_lower for reasonable in reasonable_bottle_classes)
        
        # For other objects, be more permissive
        return True
    
    def smooth_classification(self, object_key, classification):
        """Apply temporal smoothing to reduce classification flickering"""
        if object_key not in self.classification_history:
            self.classification_history[object_key] = []
        
        # Add new classification
        self.classification_history[object_key].append(classification)
        
        # Keep only recent history
        if len(self.classification_history[object_key]) > self.history_size:
            self.classification_history[object_key] = self.classification_history[object_key][-self.history_size:]
        
        # Find most common classification in recent history
        if len(self.classification_history[object_key]) >= 3:  # Need at least 3 samples
            recent_classes = [c['class'] for c in self.classification_history[object_key]]
            from collections import Counter
            most_common = Counter(recent_classes).most_common(1)[0]
            
            # If we have a clear majority (at least 60%), use that
            if most_common[1] >= len(recent_classes) * 0.6:
                # Find the most recent instance of the most common class
                for c in reversed(self.classification_history[object_key]):
                    if c['class'] == most_common[0]:
                        return c
        
        # If no clear majority or not enough samples, return the most recent
        return self.classification_history[object_key][-1]
    
    def smooth_coordinates(self, object_key, x1, y1, x2, y2):
        """Apply temporal smoothing to bounding box coordinates to reduce flickering"""
        if object_key not in self.coordinate_history:
            self.coordinate_history[object_key] = []
        
        # Add new coordinates
        coords = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        self.coordinate_history[object_key].append(coords)
        
        # Keep only recent history
        if len(self.coordinate_history[object_key]) > self.history_size:
            self.coordinate_history[object_key] = self.coordinate_history[object_key][-self.history_size:]
        
        # If we have enough samples, use weighted average coordinates (more weight to recent frames)
        if len(self.coordinate_history[object_key]) >= 3:
            total_weight = 0
            weighted_x1 = 0
            weighted_y1 = 0
            weighted_x2 = 0
            weighted_y2 = 0
            
            for i, c in enumerate(self.coordinate_history[object_key]):
                # Give more weight to recent frames (exponential decay)
                weight = 2 ** i  # Most recent gets weight 1, then 2, 4, 8, etc.
                total_weight += weight
                weighted_x1 += c['x1'] * weight
                weighted_y1 += c['y1'] * weight
                weighted_x2 += c['x2'] * weight
                weighted_y2 += c['y2'] * weight
            
            avg_x1 = weighted_x1 / total_weight
            avg_y1 = weighted_y1 / total_weight
            avg_x2 = weighted_x2 / total_weight
            avg_y2 = weighted_y2 / total_weight
            
            return int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)
        
        # If not enough samples, return current coordinates
        return x1, y1, x2, y2
    
    def smooth_confidence(self, object_key, confidence):
        """Apply temporal smoothing to confidence values to prevent on/off flickering"""
        if object_key not in self.confidence_history:
            self.confidence_history[object_key] = []
        
        # Add new confidence
        self.confidence_history[object_key].append(confidence)
        
        # Keep only recent history
        if len(self.confidence_history[object_key]) > self.history_size:
            self.confidence_history[object_key] = self.confidence_history[object_key][-self.history_size:]
        
        # If we have enough samples, use weighted average (more weight to recent frames)
        if len(self.confidence_history[object_key]) >= 3:
            total_weight = 0
            weighted_confidence = 0
            
            for i, conf in enumerate(self.confidence_history[object_key]):
                # Give more weight to recent frames (exponential decay)
                weight = 2 ** i  # Most recent gets weight 1, then 2, 4, 8, etc.
                total_weight += weight
                weighted_confidence += conf * weight
            
            avg_confidence = weighted_confidence / total_weight
            return avg_confidence
        
        # If not enough samples, return current confidence
        return confidence
    
    def get_persistent_objects(self):
        """Get objects that should still be shown based on persistence"""
        persistent_objects = []
        current_frame = self.frame_count
        
        for object_key, last_seen_frame in self.object_persistence.items():
            # If object was seen recently (within persistence_frames), include it
            if current_frame - last_seen_frame <= self.persistence_frames:
                # Get the most recent data for this object
                if object_key in self.coordinate_history and len(self.coordinate_history[object_key]) > 0:
                    coords = self.coordinate_history[object_key][-1]
                    confidence = self.confidence_history.get(object_key, [0.5])[-1] if object_key in self.confidence_history else 0.5
                    classification = self.classification_history.get(object_key, [{'class': 'unknown', 'confidence': 0.5}])[-1] if object_key in self.classification_history else {'class': 'unknown', 'confidence': 0.5}
                    
                    persistent_objects.append({
                        'object_key': object_key,
                        'coords': coords,
                        'confidence': confidence,
                        'classification': classification,
                        'frames_since_seen': current_frame - last_seen_frame
                    })
        
        return persistent_objects
    
    def run_detection(self):
        """Main detection loop with robust error handling"""
        self.logger.info("Opening camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            self.logger.error("Failed to open camera!")
            return
        
        # Camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(f"Camera initialized: {width}x{height}")
        
        print("Robust logged detection running! Press 'q' to quit")
        print("=" * 60)
        
        frame_skip = 2
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.warning("Failed to read frame")
                break
            
            self.frame_count += 1
            
            # Run detection every Nth frame
            if self.frame_count % frame_skip == 0:
                self.logger.info(f"Processing frame {self.frame_count}")
                
                try:
                    # YOLO detection
                    yolo_results = self.yolo_model(
                        frame,
                        conf=self.yolo_conf,
                        iou=0.7,
                        max_det=self.max_detections,
                        verbose=False,
                        imgsz=320,
                        device=self.device,
                        half=True,
                        agnostic_nms=True,
                        save=False,
                        show=False,
                        augment=False,
                        visualize=False
                    )
                    
                    # Robust result checking
                    if not yolo_results or len(yolo_results) == 0:
                        self.logger.info(f"No YOLO results for frame {self.frame_count}")
                        continue
                    
                    result = yolo_results[0]
                    if result is None or not hasattr(result, 'boxes'):
                        self.logger.info(f"Invalid YOLO result for frame {self.frame_count}")
                        continue
                    
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        self.logger.info(f"No boxes in YOLO result for frame {self.frame_count}")
                        continue
                    
                    self.logger.info(f"YOLO detected {len(boxes)} objects:")
                    
                    # Filter out overlapping detections of the same class
                    filtered_boxes = []
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                        confidence = float(box.conf[0].cpu())
                        class_id = int(box.cls[0].cpu())
                        class_name = self.yolo_model.names[class_id]
                        
                        # Check if this box overlaps significantly with any already selected box of the same class
                        is_duplicate = False
                        for existing_box in filtered_boxes:
                            ex_x1, ex_y1, ex_x2, ex_y2 = map(int, existing_box.xyxy[0].cpu())
                            ex_class_id = int(existing_box.cls[0].cpu())
                            
                            # Only check overlap for same class
                            if class_id == ex_class_id:
                                # Calculate intersection over union (IoU)
                                intersection_x1 = max(x1, ex_x1)
                                intersection_y1 = max(y1, ex_y1)
                                intersection_x2 = min(x2, ex_x2)
                                intersection_y2 = min(y2, ex_y2)
                                
                                if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                                    box_area = (x2 - x1) * (y2 - y1)
                                    ex_box_area = (ex_x2 - ex_x1) * (ex_y2 - ex_y1)
                                    union_area = box_area + ex_box_area - intersection_area
                                    iou = intersection_area / union_area if union_area > 0 else 0
                                    
                                    # If IoU > 0.5, consider it a duplicate
                                    if iou > 0.5:
                                        is_duplicate = True
                                        self.logger.info(f"Filtered duplicate detection: {class_name} (IoU: {iou:.3f})")
                                        break
                        
                        if not is_duplicate:
                            filtered_boxes.append(box)
                    
                    self.logger.info(f"After deduplication: {len(filtered_boxes)} objects")
                    
                    for i, box in enumerate(filtered_boxes):
                        try:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                            confidence = float(box.conf[0].cpu())
                            class_id = int(box.cls[0].cpu())
                            class_name = self.yolo_model.names[class_id]
                            
                            self.logger.info(f"  Object {i+1}: {class_name} ({confidence:.3f}) at [{x1},{y1},{x2},{y2}]")
                            
                            # Skip tiny boxes
                            if (x2 - x1) < 30 or (y2 - y1) < 30:
                                self.logger.info(f"    -> Skipped (too small)")
                                continue
                            
                            # Crop object for MobileNet classification
                            cropped_obj = frame[y1:y2, x1:x2]
                            
                            if cropped_obj.size == 0:
                                self.logger.info(f"    -> Skipped (empty crop)")
                                continue
                            
                            # MobileNet classification
                            mobilenet_result = self.classify_with_mobilenet(cropped_obj)
                            
                            # Create object key for temporal smoothing (based on position and YOLO class)
                            # Use larger grid cells to make tracking more stable
                            object_key = f"{class_name}_{x1//100}_{y1//100}"  # Larger position grouping for stability
                            
                            # Smooth coordinates to reduce bounding box flickering
                            smooth_x1, smooth_y1, smooth_x2, smooth_y2 = self.smooth_coordinates(object_key, x1, y1, x2, y2)
                            
                            # Smooth confidence to prevent on/off flickering
                            smooth_confidence = self.smooth_confidence(object_key, confidence)
                            
                            # Update persistence tracking
                            self.object_persistence[object_key] = self.frame_count
                            
                            # Debug: Log coordinate changes if significant
                            if abs(smooth_x1 - x1) > 2 or abs(smooth_y1 - y1) > 2:
                                self.logger.info(f"Coordinate smoothing: ({x1},{y1},{x2},{y2}) -> ({smooth_x1},{smooth_y1},{smooth_x2},{smooth_y2})")
                            
                            # Debug: Log confidence changes if significant
                            if abs(smooth_confidence - confidence) > 0.05:
                                self.logger.info(f"Confidence smoothing: {confidence:.3f} -> {smooth_confidence:.3f}")
                            
                            # Skip low confidence detections (use smoothed confidence for threshold check)
                            if smooth_confidence < self.yolo_conf:
                                self.logger.info(f"    -> Skipped (low smoothed confidence: {smooth_confidence:.3f})")
                                continue
                            
                            # Apply temporal smoothing if we have a MobileNet result
                            smoothed_mobilenet = None
                            if mobilenet_result:
                                # Check if the classification is reasonable before smoothing
                                if self.is_reasonable_classification(class_name, mobilenet_result['class']):
                                    smoothed_mobilenet = self.smooth_classification(object_key, mobilenet_result)
                                    self.logger.info(f"Smoothed classification: {mobilenet_result['class']} -> {smoothed_mobilenet['class']}")
                                else:
                                    self.logger.info(f"Rejected unreasonable classification: YOLO={class_name} -> MobileNet={mobilenet_result['class']} (confidence: {mobilenet_result['confidence']:.3f})")
                                    smoothed_mobilenet = None
                            
                            # Determine final result
                            final_result = None
                            color = (255, 0, 0)  # Default blue
                            thickness = 1
                            
                            if smoothed_mobilenet:
                                mobilenet_class = smoothed_mobilenet['class']
                                mobilenet_conf = smoothed_mobilenet['confidence']
                                
                                is_target, target_type = self.is_target_object(mobilenet_class)
                                
                                if is_target:
                                    final_result = {
                                        'type': 'TARGET_CONFIRMED',
                                        'class': target_type,
                                        'confidence': (smooth_confidence + mobilenet_conf) / 2
                                    }
                                    color = (0, 255, 255)  # Yellow
                                    thickness = 3
                                    self.target_objects_found += 1
                                    self.mobilenet_verified += 1
                                else:
                                    final_result = {
                                        'type': 'CONFIRMED',
                                        'class': mobilenet_class,
                                        'confidence': (smooth_confidence + mobilenet_conf) / 2
                                    }
                                    color = (0, 255, 0)  # Green
                                    thickness = 2
                                    self.mobilenet_verified += 1
                            else:
                                # MobileNet couldn't classify
                                is_target, target_type = self.is_target_object(class_name)
                                
                                if is_target:
                                    final_result = {
                                        'type': 'YOLO_TARGET',
                                        'class': target_type,
                                        'confidence': smooth_confidence
                                    }
                                    color = (255, 255, 0)  # Cyan
                                    thickness = 2
                                    self.target_objects_found += 1
                                    self.yolo_only_detections += 1
                                else:
                                    final_result = {
                                        'type': 'YOLO_ONLY',
                                        'class': class_name,
                                        'confidence': smooth_confidence
                                    }
                                    color = (255, 0, 0)  # Blue
                                    thickness = 1
                                    self.yolo_only_detections += 1
                            
                            # Log the detection
                            if final_result:
                                # Convert numpy types to Python types for JSON serialization
                                mobilenet_serializable = None
                                if smoothed_mobilenet:
                                    mobilenet_serializable = {
                                        'class': smoothed_mobilenet['class'],
                                        'confidence': float(smoothed_mobilenet['confidence'])  # Convert numpy float32 to Python float
                                    }
                                
                                log_entry = {
                                    'timestamp': datetime.now().isoformat(),
                                    'frame_number': self.frame_count,
                                    'yolo_detection': {'class': class_name, 'confidence': float(smooth_confidence)},  # Convert to Python float
                                    'mobilenet_classification': mobilenet_serializable,
                                    'final_result': {
                                        'type': final_result['type'],
                                        'class': final_result['class'],
                                        'confidence': float(final_result['confidence'])  # Convert to Python float
                                    }
                                }
                                
                                # Write to JSON log
                                self.json_log.write(json.dumps(log_entry) + '\n')
                                self.json_log.flush()
                                
                                # Log to main log file
                                if smoothed_mobilenet:
                                    mobilenet_info = f"MobileNet={smoothed_mobilenet['class']} ({smoothed_mobilenet['confidence']:.3f})"
                                else:
                                    mobilenet_info = "MobileNet=None (0.000)"
                                self.logger.info(f"Frame {self.frame_count}: YOLO={class_name} ({smooth_confidence:.3f}) | {mobilenet_info} | Final={final_result['type']} {final_result['class']}")
                                
                                # Don't draw original detection - only use persistent objects
                                # This prevents the flickering original boxes
                                self.total_detections += 1
                        
                        except Exception as e:
                            self.logger.error(f"Error processing box {i} on frame {self.frame_count}: {e}")
                            continue
                
                except Exception as e:
                    self.logger.error(f"Detection error on frame {self.frame_count}: {e}")
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
                
                # Log periodic stats
                self.logger.info(f"Stats - FPS: {fps}, Detections: {self.total_detections}, "
                               f"Targets: {self.target_objects_found}, Verified: {self.mobilenet_verified}")
            
            # Display stats
            stats_text = f"FPS:{fps} | Detections:{self.total_detections} | Targets:{self.target_objects_found} | Verified:{self.mobilenet_verified}"
            cv2.putText(frame, stats_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Legend
            cv2.putText(frame, "GRAY:Persistent Objects (No Flickering)", 
                       (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Handle persistent objects (show all objects, both current and persistent)
            persistent_objects = self.get_persistent_objects()
            for persistent_obj in persistent_objects:
                coords = persistent_obj['coords']
                confidence = persistent_obj['confidence']
                classification = persistent_obj['classification']
                frames_since_seen = persistent_obj['frames_since_seen']
                
                # Only show gray persistent boxes (no flickering)
                color = (128, 128, 128)  # Gray for all persistent objects
                thickness = 1
                label_prefix = "PERSISTENT"
                
                cv2.rectangle(frame, (coords['x1'], coords['y1']), (coords['x2'], coords['y2']), color, thickness)
                
                # Label
                label = f"{label_prefix}: {classification['class'].upper()}"
                cv2.putText(frame, label, (coords['x1'], coords['y1']-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Confidence
                conf_text = f"Y:{confidence:.2f}"
                if frames_since_seen > 0:
                    conf_text += f" ({frames_since_seen}f ago)"
                cv2.putText(frame, conf_text, (coords['x1'], coords['y1']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Display frame
            cv2.imshow('Robust Logged Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.json_log.close()
        
        # Final statistics
        session_duration = datetime.now() - self.session_start
        self.logger.info("=" * 60)
        self.logger.info("SESSION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Session duration: {session_duration}")
        self.logger.info(f"Total frames processed: {self.frame_count}")
        self.logger.info(f"Total detections: {self.total_detections}")
        self.logger.info(f"Target objects found: {self.target_objects_found}")
        self.logger.info(f"MobileNet verified: {self.mobilenet_verified}")
        self.logger.info(f"YOLO-only detections: {self.yolo_only_detections}")
        
        if self.total_detections > 0:
            verification_rate = (self.mobilenet_verified / self.total_detections) * 100
            target_rate = (self.target_objects_found / self.total_detections) * 100
            self.logger.info(f"Verification rate: {verification_rate:.1f}%")
            self.logger.info(f"Target detection rate: {target_rate:.1f}%")
        
        self.logger.info(f"Final FPS: {fps}")
        self.logger.info("=" * 60)
        
        print(f"\nLogs saved to: logs/")
        print(f"Check the log files for detailed detection information!")

def main():
    detector = RobustLoggedDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()
