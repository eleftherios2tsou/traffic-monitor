import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Vehicle classes 
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Load input video
video_path = "sample_traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Save processed video
out = cv2.VideoWriter(
    "outputs/processed_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 20,
    (frame_width, frame_height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, verbose=False)
    result = results[0]

    # Draw YOLO annotations
    annotated_frame = result.plot()

    counts = {vehicle_type: 0 for vehicle_type in VEHICLE_CLASSES}
    total_vehicles = 0

    # Count only vehicle detections
    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < 0.4:
            continue

        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name in VEHICLE_CLASSES:
            counts[class_name] += 1
            total_vehicles += 1

    # Simple traffic density estimate
    if total_vehicles < 5:
        density = "LOW"
    elif total_vehicles < 10:
        density = "MEDIUM"
    else:
        density = "HIGH"

    # Overlay analytics text
    y = 30
    cv2.putText(
        annotated_frame,
        f"Total vehicles: {total_vehicles}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    y += 35
    for vehicle_type in ["car", "truck", "bus", "motorcycle"]:
        cv2.putText(
            annotated_frame,
            f"{vehicle_type.capitalize()}: {counts[vehicle_type]}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 30

    cv2.putText(
        annotated_frame,
        f"Traffic density: {density}",
        (20, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Motorway Traffic Monitor", annotated_frame)
    out.write(annotated_frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()