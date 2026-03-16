import os
import cv2
from ultralytics import YOLO

# CONFIGURATION

# Environment variable controlling whether the video window should appear
SHOW_WINDOW = os.environ.get("SHOW_WINDOW", "true").lower() == "true"

# Input and output paths
VIDEO_PATH = "sample_traffic.mp4"
OUTPUT_PATH = "outputs/processed_output.mp4"

# Vehicle classes we want to monitor
# YOLO detects many classes, but we filter only relevant motorway vehicles
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Minimum confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.4

# LOAD MODEL

# Load pretrained YOLOv8 model
# yolov8n = nano version (small and fast)
model = YOLO("yolov8n.pt")

# VIDEO INPUT SETUP

cap = cv2.VideoCapture(VIDEO_PATH)

# Retrieve video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# If FPS is not detected properly, use a fallback
if fps == 0:
    fps = 20

# Prepare video writer for output file
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

print("Starting traffic analysis...")

# MAIN PROCESSING LOOP

while True:

    # Read next frame
    ret, frame = cap.read()

    # If video ended, exit loop
    if not ret:
        break

    # Run YOLO object detection
    results = model(frame, verbose=False)
    result = results[0]

    # Draw YOLO bounding boxes on frame
    annotated_frame = result.plot()

    # Initialize vehicle counters
    counts = {vehicle: 0 for vehicle in VEHICLE_CLASSES}
    total_vehicles = 0

    # PROCESS DETECTIONS

    for box in result.boxes:

        confidence = float(box.conf[0])

        # Ignore low confidence detections
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Get predicted class
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Only count selected vehicle classes
        if class_name in VEHICLE_CLASSES:
            counts[class_name] += 1
            total_vehicles += 1

    # TRAFFIC DENSITY ESTIMATION

    # Simple heuristic for traffic density
    if total_vehicles < 5:
        density = "LOW"
    elif total_vehicles < 10:
        density = "MEDIUM"
    else:
        density = "HIGH"

    # DRAW ANALYTICS TEXT ON FRAME

    y = 30

    # Total vehicles detected
    cv2.putText(
        annotated_frame,
        f"Total vehicles: {total_vehicles}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Display count per vehicle type
    y += 35

    for vehicle in ["car", "truck", "bus", "motorcycle"]:

        cv2.putText(
            annotated_frame,
            f"{vehicle.capitalize()}: {counts[vehicle]}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        y += 30

    # Traffic density label
    cv2.putText(
        annotated_frame,
        f"Traffic density: {density}",
        (20, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    # SAVE FRAME

    out.write(annotated_frame)


    # DISPLAY FRAME (LOCAL MODE ONLY)


    if SHOW_WINDOW:

        cv2.imshow("Motorway Traffic Monitor", annotated_frame)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break



# CLEANUP

cap.release()
out.release()

if SHOW_WINDOW:
    cv2.destroyAllWindows()

print(f"Processing complete. Output saved to {OUTPUT_PATH}")