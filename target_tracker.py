import cv2
import time
import numpy as np
from ultralytics import YOLO

# 1. Load the exported YOLOv8 Nano NCNN model directory
model = YOLO("./yolov8n_ncnn_model", task="detect")

# 2. Configure Logitech C270 Webcam
cap = cv2.VideoCapture(0)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Tracking state variables
target_id = None
missing_frames = 0
MAX_MISSING_FRAMES = 30  # Reset lock if target is lost for ~1 second

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab camera frame.")
        break

    # 3. Track Humans (class 0)
    results = model.track(
        source=frame,
        persist=True,
        classes=[0],
        imgsz=320,
        tracker="bytetrack.yaml",
        verbose=False
    )

    boxes = results[0].boxes
    target_found = False

    # Check if any objects with Track IDs were detected
    if boxes is not None and boxes.id is not None:
        # Extract coordinates and track IDs
        coords = boxes.xyxy.cpu().numpy()          # [x1, y1, x2, y2]
        track_ids = boxes.id.int().cpu().numpy()  # Track IDs

        # Lock onto the very first detected person if no target is active
        if target_id is None and len(track_ids) > 0:
            target_id = int(track_ids[0])
            print(f"[LOCK] Target acquired! Tracking ID: {target_id}")

        # Search for our locked target in current frame detections
        for box, tid in zip(coords, track_ids):
            if int(tid) == target_id:
                target_found = True
                missing_frames = 0
                
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate Bounding Box Dimensions
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                center_x = x1 + (bbox_width // 2)

                # --- 4. Evaluate Spatial Position (Left / Right / Center) ---
                if center_x < 250:
                    horizontal_pos = "LEFT"
                elif center_x > 390:
                    horizontal_pos = "RIGHT"
                else:
                    horizontal_pos = "CENTER"

                # --- 5. Evaluate Distance (Too Close / Too Far / OK) ---
                if bbox_height > 360:
                    distance_status = "TOO CLOSE"
                elif bbox_height < 130:
                    distance_status = "TOO FAR"
                else:
                    distance_status = "OK DISTANCE"

                # --- 6. Visual Overlay for Target Person ---
                # Draw Box around locked target (Cyan color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                
                # Target Status Label
                status_text = f"TARGET ID {target_id}: {horizontal_pos} | {distance_status}"
                cv2.putText(
                    frame, status_text, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )
                
                # Target Center Point
                cv2.circle(frame, (center_x, y1 + bbox_height // 2), 6, (0, 0, 255), -1)
                break

    # If target was not detected in this frame
    if not target_found and target_id is not None:
        missing_frames += 1
        cv2.putText(
            frame, f"SEARCHING TARGET ID {target_id} ({missing_frames}/{MAX_MISSING_FRAMES})", 
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
        )
        # Reset target lock if missing too long
        if missing_frames > MAX_MISSING_FRAMES:
            print(f"[RESET] Target ID {target_id} lost. Unlocking.")
            target_id = None
            missing_frames = 0

    # Draw Center Guide Lines on screen
    cv2.line(frame, (250, 0), (250, FRAME_HEIGHT), (100, 100, 100), 1)
    cv2.line(frame, (390, 0), (390, FRAME_HEIGHT), (100, 100, 100), 1)

    # Calculate & Display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Target Human Tracker - Pi 5", frame)

    # Press 'q' to quit, press 'r' to manually reset target lock
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("[MANUAL RESET] Target unlocked by user.")
        target_id = None
        missing_frames = 0

cap.release()
cv2.destroyAllWindows()
