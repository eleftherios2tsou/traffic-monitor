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

## Run Locally

```bash
pip install -r requirements.txt
python main.py