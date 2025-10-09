#!/usr/bin/env python3
"""
Simple runner script for the final object detection system.
This activates the virtual environment and runs the robust detection system.
"""

import subprocess
import sys
import os

def main():
    """Run the object detection system"""
    print("=" * 60)
    print("OBJECT DETECTION SYSTEM")
    print("=" * 60)
    print("Starting robust object detection with YOLO + MobileNetV3...")
    print("Press 'q' to quit the detection window")
    print("=" * 60)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the virtual environment
    venv_python = os.path.join(script_dir, "venv", "bin", "python")
    
    # Path to the detection script
    detection_script = os.path.join(script_dir, "robust_logged_detection.py")
    
    # Check if files exist
    if not os.path.exists(venv_python):
        print("ERROR: Virtual environment not found!")
        print("Please run: python -m venv venv")
        print("Then install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    if not os.path.exists(detection_script):
        print("ERROR: Detection script not found!")
        sys.exit(1)
    
    try:
        # Run the detection system
        subprocess.run([venv_python, detection_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Detection system failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDetection system stopped by user")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
