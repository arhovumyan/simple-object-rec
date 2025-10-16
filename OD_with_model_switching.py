#!/usr/bin/env python3

"""
Object Detection Pipeline with YOLO Model Switching
Allows you to test different YOLO models in real-time during detection.
Press number keys to switch between models.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import logging
from datetime import datetime
import os
from queue import Queue, Empty
from threading import Thread, Lock
from collections import deque

class YOLOModelSwitcher:
    def __init__(self):
        # Define all available YOLO models
        self.available_models = {
            # YOLOv8 models
            '1': ('yolov8n', 'yolov8n.pt'),
            '2': ('yolov8s', 'yolov8s.pt'),
            '3': ('yolov8m', 'yolov8m.pt'),
            
            # YOLOv11 models
            '4': ('yolov11n', 'yolo11n.pt'),
            '5': ('yolov11s', 'yolo11s.pt'),
            '6': ('yolov11m', 'yolo11m.pt'),
            '7': ('yolov11l', 'yolo11l.pt'),
            '8': ('yolov11x', 'yolo11x.pt'),
            
            # YOLOv9 models
            '9': ('yolov9n', 'yolov9n.pt'),
            '0': ('yolov9s', 'yolov9s.pt'),
            'q': ('yolov9m', 'yolov9m.pt'),
            'w': ('yolov9l', 'yolov9l.pt'),
            'e': ('yolov9x', 'yolov9x.pt'),
        }
        
        self.current_model_key = '1'  # Start with yolov8n
        self.current_model_name, self.current_model_path = self.available_models[self.current_model_key]
        self.model = None
        
        # Setup GPU
        self.setup_gpu()
        
        # Load initial model
        self.load_model()
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.detection_history = deque(maxlen=30)
        self.model_switch_time = time.time()
        
        print("YOLO Model Switcher initialized!")
        self.print_controls()
    
    def setup_gpu(self):
        """Setup GPU configuration"""
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print(f"✓ PyTorch GPU (MPS) available")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print(f"✓ PyTorch GPU (CUDA) available")
        else:
            self.device = 'cpu'
            print(f"⚠ Using CPU for PyTorch")
    
    def print_controls(self):
        """Print control instructions"""
        print("\n" + "="*60)
        print("YOLO MODEL CONTROLS:")
        print("="*60)
        print("Press number keys to switch models:")
        print("YOLOv8:  1=nano, 2=small, 3=medium")
        print("YOLOv11: 4=nano, 5=small, 6=medium, 7=large, 8=xlarge")
        print("YOLOv9:  9=nano, 0=small, q=medium, w=large, e=xlarge")
        print("Press 'ESC' to quit")
        print("="*60)
    
    def load_model(self):
        """Load the current YOLO model"""
        try:
            print(f"\nLoading {self.current_model_name}...")
            
            # Download and load model
            self.model = YOLO(self.current_model_path)
            
            # Move to device
            if self.device == 'mps':
                self.model.to(self.device)
            
            self.model_switch_time = time.time()
            print(f"✓ {self.current_model_name} loaded successfully on {self.device.upper()}")
            
        except Exception as e:
            print(f"✗ Failed to load {self.current_model_name}: {e}")
            # Fallback to yolov8n if loading fails
            if self.current_model_key != '1':
                self.current_model_key = '1'
                self.current_model_name, self.current_model_path = self.available_models[self.current_model_key]
                self.load_model()
    
    def switch_model(self, key):
        """Switch to a different YOLO model"""
        if key in self.available_models:
            if key != self.current_model_key:
                self.current_model_key = key
                self.current_model_name, self.current_model_path = self.available_models[key]
                self.load_model()
                return True
        return False
    
    def detect_objects(self, frame):
        """Detect objects with current model"""
        try:
            results = self.model(
                frame,
                conf=0.4,
                iou=0.5,
                max_det=10,
                verbose=False,
                device=self.device
            )
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                        confidence = float(box.conf[0].cpu())
                        class_id = int(box.cls[0].cpu())
                        class_name = self.model.names[class_id]
                        
                        # Filter small objects
                        if (x2 - x1) < 15 or (y2 - y1) < 15:
                            continue
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_name': class_name,
                            'class_id': class_id
                        })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw detections and info on frame"""
        height, width = frame.shape[:2]
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Color based on confidence
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw model info
        model_time = time.time() - self.model_switch_time
        cv2.putText(frame, f"Model: {self.current_model_name.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Device: {self.device.upper()}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Model runtime: {model_time:.1f}s", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw FPS
        if len(self.fps_history) > 0:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                       (width-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw controls reminder
        cv2.putText(frame, "Press 1-9,0,q,w,e to switch models", 
                   (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ESC to quit", 
                   (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main detection loop with model switching"""
        print("Opening camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            print("Failed to open camera!")
            return
        
        # Camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {width}x{height}")
        
        print(f"\nStarting detection with {self.current_model_name}...")
        print("Use number keys to switch models in real-time!")
        
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Run detection
                detections = self.detect_objects(frame)
                
                # Update detection history
                self.detection_history.append(len(detections))
                
                # Draw everything
                frame = self.draw_detections(frame, detections)
                
                # Calculate FPS
                if current_time - fps_start_time >= 1.0:
                    fps = frame_count
                    self.fps_history.append(fps)
                    frame_count = 0
                    fps_start_time = current_time
                
                # Display frame
                cv2.imshow('YOLO Model Switcher', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif chr(key) in self.available_models:
                    if self.switch_model(chr(key)):
                        print(f"Switched to {self.current_model_name}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Final performance summary
            if self.fps_history:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                avg_detections = sum(self.detection_history) / len(self.detection_history) if self.detection_history else 0
                
                print(f"\n{'='*60}")
                print("FINAL PERFORMANCE SUMMARY")
                print(f"{'='*60}")
                print(f"Final model: {self.current_model_name}")
                print(f"Average FPS: {avg_fps:.2f}")
                print(f"Average detections per frame: {avg_detections:.2f}")
                print(f"Device: {self.device.upper()}")
                print(f"{'='*60}")

def main():
    """Main function"""
    try:
        switcher = YOLOModelSwitcher()
        switcher.run()
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
