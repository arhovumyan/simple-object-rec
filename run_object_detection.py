#!/usr/bin/env python3

"""
Object Detection and Classification Pipeline
First uses YOLOv8 for object detection, then MobileNetV3 for classification
to determine if detected objects are what we're looking for.
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

class ObjectDetectionPipeline:
    def __init__(self):
        print("=" * 60)
        print("OBJECT DETECTION & CLASSIFICATION PIPELINE")
        print("=" * 60)
        print("Stage 1: YOLOv8 Object Detection")
        print("Stage 2: MobileNetV3 Classification")
        print("=" * 60)
        
        # Setup logging
        self.setup_logging()
        
        # Setup devices
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"YOLO Device: {self.device.upper()}")
        
        # Load YOLOv8 model
        print("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')
        if self.device == 'mps':
            self.yolo_model.to(self.device)
        
        # Load MobileNetV3 model
        print("Loading MobileNetV3 model...")
        tf.config.set_visible_devices([], 'GPU')  # Use CPU for TensorFlow
        self.mobilenet_model = MobileNetV3Small(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=True
        )
        
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
        
        # Simplified detection - no persistence to avoid multiple boxes
        self.object_counter = 0  # Simple counter for unique object IDs
        
        self.logger.info("Models loaded successfully!")
        self.logger.info(f"Target objects: {list(self.target_objects.keys())}")
    
    def setup_logging(self):
        """Setup logging system"""
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger = logging.getLogger('ObjectDetectionPipeline')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(f'logs/object_detection_{timestamp}.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        print(f"Logging to: logs/object_detection_{timestamp}.log")
    
    def detect_objects_yolo(self, frame):
        """Stage 1: Use YOLOv8 for object detection"""
        try:
            # Run YOLO detection
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
                
                # First pass: collect all detections
                raw_detections = []
                for i, box in enumerate(boxes):
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                    confidence = float(box.conf[0].cpu())
                    class_id = int(box.cls[0].cpu())
                    class_name = self.yolo_model.names[class_id]
                    
                    # Skip very small detections
                    if (x2 - x1) < 30 or (y2 - y1) < 30:
                        continue
                    
                    raw_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_name': class_name,
                        'class_id': class_id
                    })
                
                # Second pass: deduplicate overlapping detections
                detections = self.deduplicate_detections(raw_detections)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
            return []
    
    def deduplicate_detections(self, raw_detections):
        """Remove overlapping detections of the same class"""
        if not raw_detections:
            return []
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(raw_detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        
        for detection in sorted_detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Check if this detection overlaps significantly with ANY already accepted detection
            is_duplicate = False
            for accepted in filtered_detections:
                ax1, ay1, ax2, ay2 = accepted['bbox']
                
                # Calculate intersection over union (IoU)
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
                    
                    # Much more aggressive IoU threshold - any significant overlap
                    if iou > 0.3:
                        is_duplicate = True
                        self.logger.info(f"Filtered duplicate {class_name} detection (IoU: {iou:.3f})")
                        break
            
            if not is_duplicate:
                filtered_detections.append(detection)
                self.logger.info(f"YOLO detected: {class_name} ({confidence:.3f}) at [{x1},{y1},{x2},{y2}]")
                
                # Limit to maximum 3 detections to prevent clutter
                if len(filtered_detections) >= 3:
                    self.logger.info("Reached maximum detection limit (3)")
                    break
        
        return filtered_detections
    
    def classify_with_mobilenet(self, cropped_image):
        """Stage 2: Use MobileNetV3 for classification"""
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
            decoded_predictions = decode_predictions(predictions, top=3)[0]
            
            # Log top predictions
            self.logger.info("MobileNetV3 top predictions:")
            for i, (class_id, class_name, confidence) in enumerate(decoded_predictions):
                self.logger.info(f"  {i+1}. {class_name}: {confidence:.3f}")
            
            # Return the top prediction
            if decoded_predictions:
                top_prediction = decoded_predictions[0]
                return {
                    'class': top_prediction[1],
                    'confidence': top_prediction[2],
                    'all_predictions': decoded_predictions
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"MobileNetV3 classification error: {e}")
            return None
    
    def is_target_object(self, class_name):
        """Check if the classified object is one of our target objects"""
        class_lower = class_name.lower()
        
        for target_type, keywords in self.target_objects.items():
            for keyword in keywords:
                if keyword.lower() in class_lower:
                    return True, target_type
        
        return False, None
    
    def smooth_coordinates(self, object_key, x1, y1, x2, y2):
        """Apply temporal smoothing to bounding box coordinates"""
        if object_key not in self.coordinate_history:
            self.coordinate_history[object_key] = []
        
        # Add new coordinates
        coords = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        self.coordinate_history[object_key].append(coords)
        
        # Keep only recent history
        if len(self.coordinate_history[object_key]) > self.history_size:
            self.coordinate_history[object_key] = self.coordinate_history[object_key][-self.history_size:]
        
        # If we have enough samples, use weighted average (more weight to recent frames)
        if len(self.coordinate_history[object_key]) >= 3:
            total_weight = 0
            weighted_x1 = 0
            weighted_y1 = 0
            weighted_x2 = 0
            weighted_y2 = 0
            
            for i, c in enumerate(self.coordinate_history[object_key]):
                # Give more weight to recent frames
                weight = 2 ** i
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
        """Apply temporal smoothing to confidence values"""
        if object_key not in self.confidence_history:
            self.confidence_history[object_key] = []
        
        # Add new confidence
        self.confidence_history[object_key].append(confidence)
        
        # Keep only recent history
        if len(self.confidence_history[object_key]) > self.history_size:
            self.confidence_history[object_key] = self.confidence_history[object_key][-self.history_size:]
        
        # If we have enough samples, use weighted average
        if len(self.confidence_history[object_key]) >= 3:
            total_weight = 0
            weighted_confidence = 0
            
            for i, conf in enumerate(self.confidence_history[object_key]):
                weight = 2 ** i
                total_weight += weight
                weighted_confidence += conf * weight
            
            return weighted_confidence / total_weight
        
        # If not enough samples, return current confidence
        return confidence
    
    def smooth_classification(self, object_key, result):
        """Apply temporal smoothing to classification results"""
        if object_key not in self.object_history:
            self.object_history[object_key] = []
        
        # Add new result
        self.object_history[object_key].append(result)
        
        # Keep only recent history
        if len(self.object_history[object_key]) > self.history_size:
            self.object_history[object_key] = self.object_history[object_key][-self.history_size:]
        
        # If we have enough samples, find the most common classification
        if len(self.object_history[object_key]) >= 3:
            # Count classifications
            classifications = [r['is_target'] for r in self.object_history[object_key]]
            target_count = sum(classifications)
            
            # If majority are targets, use the most recent target result
            if target_count >= len(classifications) * 0.6:
                for r in reversed(self.object_history[object_key]):
                    if r['is_target']:
                        return r
            else:
                # If majority are non-targets, use the most recent non-target result
                for r in reversed(self.object_history[object_key]):
                    if not r['is_target']:
                        return r
        
        # If not enough samples, return the most recent
        return self.object_history[object_key][-1]
    
    def get_persistent_objects(self, current_frame):
        """Get objects that should still be shown based on persistence"""
        persistent_objects = []
        
        for object_key, last_seen_frame in self.object_persistence.items():
            # If object was seen recently (within persistence_frames), include it
            if current_frame - last_seen_frame <= self.persistence_frames:
                # Get the most recent data for this object
                if object_key in self.coordinate_history and len(self.coordinate_history[object_key]) > 0:
                    coords = self.coordinate_history[object_key][-1]
                    confidence = self.confidence_history.get(object_key, [0.5])[-1] if object_key in self.confidence_history else 0.5
                    result = self.object_history.get(object_key, [{'is_target': False, 'target_type': None}])[-1] if object_key in self.object_history else {'is_target': False, 'target_type': None}
                    
                    persistent_objects.append({
                        'object_key': object_key,
                        'coords': coords,
                        'confidence': confidence,
                        'result': result,
                        'frames_since_seen': current_frame - last_seen_frame
                    })
        
        return persistent_objects
    
    def process_detection(self, frame, detection, frame_count):
        """Process a single detection through the pipeline - simplified version"""
        x1, y1, x2, y2 = detection['bbox']
        yolo_class = detection['class_name']
        yolo_confidence = detection['confidence']
        
        # Crop the detected object
        cropped_obj = frame[y1:y2, x1:x2]
        
        if cropped_obj.size == 0:
            return None
        
        # Stage 2: Classify with MobileNetV3
        self.classifications_performed += 1
        classification_result = self.classify_with_mobilenet(cropped_obj)
        
        result = {
            'bbox': (x1, y1, x2, y2),
            'yolo_class': yolo_class,
            'yolo_confidence': yolo_confidence,
            'mobilenet_classification': classification_result,
            'is_target': False,
            'target_type': None,
            'final_confidence': yolo_confidence
        }
        
        # Determine if this is what we're looking for
        if classification_result:
            mobilenet_class = classification_result['class']
            mobilenet_confidence = classification_result['confidence']
            
            is_target, target_type = self.is_target_object(mobilenet_class)
            
            if is_target:
                result['is_target'] = True
                result['target_type'] = target_type
                result['final_confidence'] = (yolo_confidence + mobilenet_confidence) / 2
                self.target_confirmations += 1
                self.logger.info(f"TARGET CONFIRMED: {target_type} (YOLO: {yolo_class}, MobileNet: {mobilenet_class})")
            else:
                self.logger.info(f"Not a target: YOLO={yolo_class}, MobileNet={mobilenet_class}")
        
        # Return result directly - no smoothing or persistence
        return result
    
    def draw_persistent_results(self, frame, persistent_objects):
        """Draw persistent detection results on frame (anti-flickering)"""
        for persistent_obj in persistent_objects:
            coords = persistent_obj['coords']
            confidence = persistent_obj['confidence']
            result = persistent_obj['result']
            frames_since_seen = persistent_obj['frames_since_seen']
            
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            # Choose color based on whether it's a target
            if result['is_target']:
                color = (0, 255, 0)  # Green for targets
                thickness = 3
                label = f"TARGET: {result['target_type'].upper()}"
            else:
                color = (0, 0, 255)  # Red for non-targets
                thickness = 2
                label = f"NON-TARGET: {result['yolo_class']}"
            
            # Make boxes slightly transparent if they're old detections
            if frames_since_seen > 0:
                alpha = 0.7  # Make older detections slightly transparent
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                # Draw bounding box for current detections
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw confidence
            conf_text = f"Conf: {confidence:.2f}"
            if frames_since_seen > 0:
                conf_text += f" ({frames_since_seen}f ago)"
            cv2.putText(frame, conf_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw MobileNet classification if available
            if result.get('mobilenet_classification'):
                mobilenet_class = result['mobilenet_classification']['class']
                mobilenet_conf = result['mobilenet_classification']['confidence']
                mobile_text = f"MobileNet: {mobilenet_class} ({mobilenet_conf:.2f})"
                cv2.putText(frame, mobile_text, (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_current_detections(self, frame, detections):
        """Draw only current detections (no persistence to avoid multiple boxes)"""
        for result in detections:
            x1, y1, x2, y2 = result['bbox']
            
            # Choose color based on whether it's a target
            if result['is_target']:
                color = (0, 255, 0)  # Green for targets
                thickness = 3
                label = f"TARGET: {result['target_type'].upper()}"
            else:
                color = (0, 0, 255)  # Red for non-targets
                thickness = 2
                label = f"NON-TARGET: {result['yolo_class']}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw confidence
            confidence = result['final_confidence']
            conf_text = f"Conf: {confidence:.2f}"
            cv2.putText(frame, conf_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw MobileNet classification if available
            if result.get('mobilenet_classification'):
                mobilenet_class = result['mobilenet_classification']['class']
                mobilenet_conf = result['mobilenet_classification']['confidence']
                mobile_text = f"MobileNet: {mobilenet_class} ({mobilenet_conf:.2f})"
                cv2.putText(frame, mobile_text, (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_results(self, frame, results):
        """Draw detection results on frame (legacy method - kept for compatibility)"""
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            
            # Choose color based on whether it's a target
            if result['is_target']:
                color = (0, 255, 0)  # Green for targets
                thickness = 3
                label = f"TARGET: {result['target_type'].upper()}"
            else:
                color = (0, 0, 255)  # Red for non-targets
                thickness = 2
                label = f"NON-TARGET: {result['yolo_class']}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw confidence
            conf_text = f"Conf: {result['final_confidence']:.2f}"
            cv2.putText(frame, conf_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw MobileNet classification if available
            if result['mobilenet_classification']:
                mobilenet_class = result['mobilenet_classification']['class']
                mobilenet_conf = result['mobilenet_classification']['confidence']
                mobile_text = f"MobileNet: {mobilenet_class} ({mobilenet_conf:.2f})"
                cv2.putText(frame, mobile_text, (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def run_detection(self):
        """Main detection loop"""
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
        
        print("Detection pipeline running! Press 'q' to quit")
        print("=" * 60)
        
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        # Detection timing control
        last_detection_time = 0
        detection_interval = 0.1  # 100ms = 10 times per second
        current_detections = []  # Store current detections for drawing
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Run detection exactly 10 times per second (every 100ms)
            if current_time - last_detection_time >= detection_interval:
                # Stage 1: YOLOv8 Detection
                detections = self.detect_objects_yolo(frame)
                
                # Process each detection and store for drawing
                current_detections = []  # Reset current detections
                for detection in detections:
                    result = self.process_detection(frame, detection, frame_count)
                    if result:
                        current_detections.append(result)
                        self.total_detections += 1
                        if result['is_target']:
                            self.target_objects_found += 1
                
                last_detection_time = current_time
            
            # Draw only current detections (no persistence to avoid multiple boxes)
            self.draw_current_detections(frame, current_detections)
            
            # Calculate and display FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
                
                # Display stats
                stats_text = f"FPS: {fps} | Detections: {self.total_detections} | Targets: {self.target_objects_found}"
                cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Legend
            cv2.putText(frame, "GREEN: Target Objects | RED: Non-Target Objects (Anti-Flickering)", 
                       (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Object Detection & Classification Pipeline', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "=" * 60)
        print("DETECTION PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Total detections: {self.total_detections}")
        print(f"Target objects found: {self.target_objects_found}")
        print(f"Classifications performed: {self.classifications_performed}")
        print(f"Target confirmations: {self.target_confirmations}")
        
        if self.total_detections > 0:
            target_rate = (self.target_objects_found / self.total_detections) * 100
            print(f"Target detection rate: {target_rate:.1f}%")
        
        print("=" * 60)

def main():
    """Main function"""
    try:
        detector = ObjectDetectionPipeline()
        detector.run_detection()
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
