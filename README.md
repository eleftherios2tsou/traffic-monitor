# Motorway Traffic Monitoring Prototype

A computer vision prototype for monitoring motorway traffic using YOLOv8 and OpenCV.

## Overview

This project detects and classifies vehicles in a motorway video feed and provides simple traffic analytics in real time.

It was built as a learning project to explore object detection, computer vision pipelines, and video-based ML inference.

## Features

- Real-time vehicle detection
- Vehicle classification (car, truck, bus, motorcycle)
- On-screen vehicle counts
- Traffic density estimation
- Output video saving

## Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy
- Docker

## How It Works

1. Load a traffic video
2. Run YOLOv8 object detection on each frame
3. Filter for vehicle-related classes
4. Count detections by type
5. Estimate traffic density
6. Save and display the annotated result

## Evaluation

A simple count-based evaluation was performed on a sample motorway video.

### Ground Truth
- Car: 28
- Truck: 2
- Bus: 0
- Motorcycle: 0

### Predicted Crossing Counts
- Car: 28
- Truck: 1
- Bus: 1
- Motorcycle: 0

### Observations
- All 30 vehicles were detected
- One truck was misclassified as a bus
- This indicates strong detection performance but a minor classification confusion between visually similar heavy vehicles

### Note
This project currently uses aggregate count-based evaluation rather than full object detection metrics such as mAP, since the sample video was not frame-by-frame annotated.

## Run Locally

```bash
pip install -r requirements.txt
python main.py