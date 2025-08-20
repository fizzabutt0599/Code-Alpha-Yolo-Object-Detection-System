# Code-Alpha-Yolo-Object-Detection-System
# Code-Alpha YOLO Object Detection System

## Overview
This project is a **real-time video object detection system** using **YOLOv8** and Python. Users can upload a video, select an object class (or all objects), and the system will detect and track objects with bounding boxes and confidence scores.

## Features
- Upload video files in MP4 format.
- Detect specific object classes or all objects in the video.
- Display detected objects with unique bounding box colors.
- Start and stop detection anytime.
- Simple GUI using **Tkinter**.
- Multi-threaded processing for smooth video playback.

## Requirements
- Python 3.10+
- OpenCV
- Pillow
- ultralytics (YOLOv8)
- Tkinter (built-in with Python)
- threading (built-in)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/fizzabutt0599/Code-Alpha-Yolo-Object-Detection-System.git

**2.Navigate to project folder:**
bash:
cd Code-Alpha-Yolo-Object-Detection-System

**3.Create a virtual environment:**
bash:
python -m venv .venv

**4.Activate the environment:**
powershell:
.\.venv\Scripts\Activate.ps1

**5.Install dependencies:**
bash:
pip install -r requirements.txt

**6.Run the GUI:**
bash/powershell:
pip install ultralytics
python webcam_yolo_tracking.py

