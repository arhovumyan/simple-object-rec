#!/usr/bin/env python3

"""
Detection methods for ROS2 Object Detection Pipeline
Contains all the detection, tracking, and classification methods
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
import time
import logging
from queue import Queue, Empty
from collections import deque
from scipy.optimize import linear_sum_assignment

class DetectionMethods:
    """Container class for all detection methods"""
    
    def __init__(self, logger, yolo_model, mobilenet_model, device, parameters):
        self.logger = logger
        self.yolo_model = yolo_model
        self.mobilenet_model = mobilenet_model
        self.device = device
        
        # Parameters
        self.yolo_confidence = parameters['yolo_confidence']
        self.mobilenet_confidence = parameters['mobilenet_confidence']
        self.max_detections = parameters['max_detections']
        self.min_box_size = parameters['min_box_size']
        self.multi_scale_detection = parameters['multi_scale_detection']
        self.enhance_small_objects = parameters['enhance_small_objects']
        self.ultra_small_detection = parameters['ultra_small_detection']
        self.motion_threshold = parameters['motion_threshold']
        self.base_detection_interval = parameters['base_detection_interval']
        self.max_detection_interval = parameters['max_detection_interval']
        self.max_track_age = parameters['max_track_age']
        self.min_track_hits = parameters['min_track_hits']
        self.tracking_threshold = parameters['tracking_threshold']
        self.small_object_tracking = parameters['small_object_tracking']
        
        # Target objects
        self.target_objects = {
            'tent': ['tent', 'camping_tent', 'pup_tent', 'canvas_tent', 'backpacking_tent', 'dome_tent'],
            'mannequin': ['mannequin', 'dummy', 'model', 'display_model', 'store_dummy', 'fashion_model']
        }
        
        # Motion detection
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)
        self.static_frame_count = 0
        self.motion_detected = False
        self.current_detection_interval = self.base_detection_interval
        
        # Tracking
        self.tracked_objects = {}
        self.next_track_id = 1
        
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
    
    def detect_objects_yolo(self, frame):
        """Stage 1: Use YOLOv8 for object detection with multi-scale support"""
        try:
            all_detections = []
            
            if self.multi_scale_detection:
                height, width = frame.shape[:2]
                
                # Scale factors for different object sizes
                scale_factors = [1.0, 1.5]
                confidence_factors = [1.0, 0.8]
                
                for i, (scale, conf_factor) in enumerate(zip(scale_factors, confidence_factors)):
                    if scale == 1.0:
                        # Original scale
                        results = self.yolo_model(
                            frame,
                            conf=self.yolo_confidence * conf_factor,
                            iou=0.5,
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
                            iou=0.5,
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
                                detection['scale'] = scale
                                detection['confidence'] *= 0.95
                        
                        all_detections.extend(scale_detections)
                        
            else:
                # Single scale detection
                results = self.yolo_model(
                    frame,
                    conf=self.yolo_confidence,
                    iou=0.5,
                    max_det=self.max_detections,
                    verbose=False,
                    device=self.device
                )
                if results and len(results) > 0:
                    all_detections = self._process_yolo_results(results, frame.shape)
            
            # Remove duplicates and merge results
            detections = self._merge_detections_aggressive(all_detections)
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
    
    def _filter_detections_by_size(self, detections):
        """Filter detections to keep only reasonable sizes"""
        filtered = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Keep detections that are at least minimum size
            if (width >= self.min_box_size and height >= self.min_box_size and
                width < 800 and height < 800):
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
                    iou_threshold = 0.2
                elif box_area < 1600:  # Small objects
                    iou_threshold = 0.3
                else:  # Larger objects
                    iou_threshold = 0.4
                
                # Check for overlap regardless of class
                if iou > iou_threshold:
                    # If same class, merge and keep higher confidence
                    if detection['class_name'] == merged['class_name']:
                        if detection['confidence'] > merged['confidence']:
                            merged['bbox'] = detection['bbox']
                            merged['confidence'] = detection['confidence']
                    else:
                        # Different classes but overlapping - keep the one with higher confidence
                        if detection['confidence'] > merged['confidence']:
                            merged_detections.remove(merged)
                            merged_detections.append(detection)
                            merged = detection
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_detections.append(detection)
                
                # Limit total detections for performance
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
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced1 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            enhanced_versions.append(enhanced1)
            
            # Version 2: Gamma correction for better contrast
            gamma = 1.2
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
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            tophat_color = cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)
            enhanced5 = cv2.add(frame, tophat_color)
            enhanced_versions.append(enhanced5)
            
            # Combine all enhanced versions using weighted average
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]
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
                    self.current_detection_interval * 1.1,
                    self.max_detection_interval
                )
            elif self.static_frame_count > 10:  # After 1 second of no motion
                self.current_detection_interval = min(
                    self.current_detection_interval * 1.05,
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
        
        # Calculate cost matrix
        detections_len = len(detections)
        tracks_len = len(self.tracked_objects)
        
        if detections_len == 0:
            return list(self.tracked_objects.keys())
        
        cost_matrix = np.zeros((tracks_len, detections_len))
        track_ids = list(self.tracked_objects.keys())
        
        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracked_objects[track_id]['bbox']
            
            for j, detection in enumerate(detections):
                detection_bbox = detection['bbox']
                
                # Calculate IoU
                iou = self.calculate_iou(track_bbox, detection_bbox)
                cost_matrix[i, j] = 1.0 - iou
        
        # Hungarian algorithm for optimal assignment
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Update matched tracks
        matched_tracks = set()
        matched_detections = set()
        
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            track_id = track_ids[track_idx]
            detection = detections[detection_idx]
            
            # Only update if cost is below threshold
            if cost_matrix[track_idx, detection_idx] < (1.0 - self.tracking_threshold):
                self.tracked_objects[track_id]['bbox'] = detection['bbox']
                self.tracked_objects[track_id]['yolo_class'] = detection['class_name']
                self.tracked_objects[track_id]['yolo_confidence'] = detection['confidence']
                self.tracked_objects[track_id]['hits'] += 1
                self.tracked_objects[track_id]['age'] = 0
                self.tracked_objects[track_id]['last_seen'] = frame_count
                
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
    
    def queue_for_classification(self, frame, track_id, track_info, timestamp, classification_queue):
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
            if not classification_queue.full():
                classification_queue.put(task, block=False)
                self.queue_stats['total_queued'] += 1
                
                # Track queue size
                current_size = classification_queue.qsize()
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
            
            # Stack into batch
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
    
    def is_target_object(self, class_name):
        """Check if the classified object is one of our target objects"""
        class_lower = class_name.lower()
        
        for target_type, keywords in self.target_objects.items():
            for keyword in keywords:
                if keyword.lower() in class_lower:
                    return True, target_type
        
        return False, None
    
    def get_processed_results(self, results_queue):
        """Get all processed classification results from queue"""
        results = []
        
        try:
            while not results_queue.empty():
                result = results_queue.get_nowait()
                
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
        used_label_positions = []
        
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
