import os
import cv2
from ultralytics import YOLO

# CONFIGURATION

SHOW_WINDOW = os.environ.get("SHOW_WINDOW", "true").lower() == "true"

VIDEO_PATH = "sample_traffic.mp4"
OUTPUT_PATH = "outputs/processed_output.mp4"

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
CONFIDENCE_THRESHOLD = 0.4
# TRUE VALUES
GROUND_TRUTH = {
    "car": 28,
    "truck": 2,
    "bus": 0,
    "motorcycle": 0
}
# Horizontal counting line position
LINE_Y = 250

# LOAD MODEL

model = YOLO("yolov8n.pt")

# VIDEO SETUP

cap = cv2.VideoCapture(VIDEO_PATH)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 20

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

# TRACKING STATE

# Stores previous y-center position for each tracked object ID
previous_positions = {}

# Stores IDs already counted after crossing the line
counted_ids = set()

# Stores total unique counts by vehicle type
crossing_counts = {
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0
}

print("Starting traffic analysis with tracking and line counting...")

# MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking instead of plain detection
    results = model.track(frame, persist=True, verbose=False)
    result = results[0]

    annotated_frame = frame.copy()

    # Draw counting line
    cv2.line(
        annotated_frame,
        (0, LINE_Y),
        (frame_width, LINE_Y),
        (0, 255, 255),
        2
    )

    # Current frame visible counts
    visible_counts = {vehicle: 0 for vehicle in VEHICLE_CLASSES}
    total_visible = 0

    # If no detections, continue safely
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes
        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name not in VEHICLE_CLASSES:
                continue

            visible_counts[class_name] += 1
            total_visible += 1

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Center point of the object
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with class + tracking ID
            label = f"{class_name} ID:{track_id} {confidence:.2f}"
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Draw center point
            cv2.circle(annotated_frame, (center_x, center_y), 4, (0, 0, 255), -1)

            # Check whether the object crossed the line
            if track_id in previous_positions:
                prev_y = previous_positions[track_id]

                crossed_downward = prev_y < LINE_Y <= center_y
                crossed_upward = prev_y > LINE_Y >= center_y

                # Count only once
                if (crossed_downward or crossed_upward) and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    crossing_counts[class_name] += 1

            # Update latest center position
            previous_positions[track_id] = center_y

    # TRAFFIC DENSITY

    if total_visible < 5:
        density = "LOW"
    elif total_visible < 12:
        density = "MEDIUM"
    else:
        density = "HIGH"

    # OVERLAY TEXT

    y = 30

    cv2.putText(
        annotated_frame,
        f"Visible vehicles: {total_visible}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    y += 35
    cv2.putText(
        annotated_frame,
        f"Traffic density: {density}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    y += 40
    cv2.putText(
        annotated_frame,
        "Crossing counts:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    y += 35
    for vehicle in ["car", "truck", "bus", "motorcycle"]:
        cv2.putText(
            annotated_frame,
            f"{vehicle.capitalize()}: {crossing_counts[vehicle]}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 30

    # Save frame
    out.write(annotated_frame)

    # Optional display
    if SHOW_WINDOW:
        cv2.imshow("Motorway Traffic Monitor", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# CLEANUP

cap.release()
out.release()

if SHOW_WINDOW:
    cv2.destroyAllWindows()

print("Processing complete.")
print(f"Output saved to: {OUTPUT_PATH}")
print("Final crossing counts:", crossing_counts)
print("\nEvaluation Summary")
print("-" * 30)

total_true = sum(GROUND_TRUTH.values())
total_pred = sum(crossing_counts.values())

correct_by_count = 0
absolute_error = 0

for vehicle in ["car", "truck", "bus", "motorcycle"]:
    true_count = GROUND_TRUTH.get(vehicle, 0)
    pred_count = crossing_counts.get(vehicle, 0)
    error = pred_count - true_count
    absolute_error += abs(error)

    print(
        f"{vehicle.capitalize():<12} | Ground Truth: {true_count:<3} | "
        f"Predicted: {pred_count:<3} | Error: {error:+d}"
    )

    # crude overlap count for approximate correctness
    correct_by_count += min(true_count, pred_count)

classification_accuracy = correct_by_count / total_true if total_true > 0 else 0

print("-" * 30)
print(f"Total ground truth vehicles: {total_true}")
print(f"Total predicted vehicles:    {total_pred}")
print(f"Approx. correct by count:    {correct_by_count}")
print(f"Approx. classification accuracy: {classification_accuracy:.2%}")
print(f"Total absolute count error:  {absolute_error}")