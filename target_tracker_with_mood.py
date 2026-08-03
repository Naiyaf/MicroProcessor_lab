import cv2
import time
import numpy as np
import threading
from ultralytics import YOLO

# Uncomment this when you download the TFLite model:
# from tflite_runtime.interpreter import Interpreter 

# --- 1. THREADING SETUP FOR MOOD TRACKING ---
latest_face_crop = None
current_mood = "NEUTRAL"
lock = threading.Lock()

def emotion_worker():
    """Background thread to process facial emotions without lagging YOLO."""
    global latest_face_crop, current_mood
    
    # --- TFLite Model Setup Placeholder ---
    # interpreter = Interpreter(model_path="fer2013_mini.tflite")
    # interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    while True:
        # Safely grab the latest face crop from the main thread
        with lock:
            face_img = latest_face_crop
            latest_face_crop = None  # Reset after grabbing

        if face_img is not None and face_img.size > 0:
            try:
                # Preprocess the image (e.g., resize to 48x48 Grayscale for FER-2013)
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (48, 48))
                
                # --- ACTUAL TFLITE INFERENCE GOES HERE ---
                # input_data = np.expand_dims(np.expand_dims(resized, -1), 0).astype(np.float32) / 255.0
                # interpreter.set_tensor(input_details[0]['index'], input_data)
                # interpreter.invoke()
                # preds = interpreter.get_tensor(output_details[0]['index'])
                # mood_idx = np.argmax(preds)
                # detected_mood = emotions[mood_idx]
                
                # --- SIMULATION (For immediate FPS testing) ---
                time.sleep(0.05) # Simulate 50ms model processing time
                detected_mood = "HAPPY" # Simulated output
                
                with lock:
                    current_mood = detected_mood
                    
            except Exception as e:
                print(f"Emotion thread error: {e}")
        else:
            time.sleep(0.05) # Sleep if no face is available to save CPU cycles

# Start the background emotion thread
emotion_thread = threading.Thread(target=emotion_worker, daemon=True)
emotion_thread.start()


# --- 2. MAIN TRACKING SCRIPT ---
# Load the exported YOLOv8 Nano NCNN model directory
model = YOLO("./yolov8n_ncnn_model", task="detect")

# Configure Logitech C270 Webcam
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

    # Track Humans (class 0)
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
        coords = boxes.xyxy.cpu().numpy()          # [x1, y1, x2, y2]
        track_ids = boxes.id.int().cpu().numpy()  # Track IDs

        if target_id is None and len(track_ids) > 0:
            target_id = int(track_ids[0])
            print(f"[LOCK] Target acquired! Tracking ID: {target_id}")

        for box, tid in zip(coords, track_ids):
            if int(tid) == target_id:
                target_found = True
                missing_frames = 0
                
                x1, y1, x2, y2 = map(int, box)
                
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                center_x = x1 + (bbox_width // 2)

                # --- EXTRACT HEAD FOR MOOD TRACKING ---
                # Isolate the top 30% of the bounding box
                head_y2 = y1 + int(bbox_height * 0.3)
                
                # Clamp coordinates so they don't exceed frame boundaries
                head_y2 = min(max(head_y2, 0), FRAME_HEIGHT)
                y1_cl = min(max(y1, 0), FRAME_HEIGHT)
                x1_cl = min(max(x1, 0), FRAME_WIDTH)
                x2_cl = min(max(x2, 0), FRAME_WIDTH)
                
                face_crop = frame[y1_cl:head_y2, x1_cl:x2_cl]
                
                # Safely send this crop to the background thread
                with lock:
                    latest_face_crop = face_crop.copy() if face_crop.size > 0 else None

                # Evaluate Spatial Position
                if center_x < 250:
                    horizontal_pos = "LEFT"
                elif center_x > 390:
                    horizontal_pos = "RIGHT"
                else:
                    horizontal_pos = "CENTER"

                # Evaluate Distance
                if bbox_height > 360:
                    distance_status = "TOO CLOSE"
                elif bbox_height < 130:
                    distance_status = "TOO FAR"
                else:
                    distance_status = "OK DISTANCE"

                # --- VISUAL OVERLAYS ---
                # Draw Box around locked target
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                
                # Draw a thin green box showing exactly what the AI sees for mood tracking
                cv2.rectangle(frame, (x1_cl, y1_cl), (x2_cl, head_y2), (0, 255, 0), 1)
                
                # Fetch the current mood from the background thread safely
                with lock:
                    display_mood = current_mood
                
                # Labels
                status_text = f"ID {target_id}: {horizontal_pos} | {distance_status}"
                mood_text = f"MOOD: {display_mood}"
                
                cv2.putText(frame, status_text, (x1, max(y1 - 25, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, mood_text, (x1, max(y1 - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.circle(frame, (center_x, y1 + bbox_height // 2), 6, (0, 0, 255), -1)
                break

    if not target_found and target_id is not None:
        missing_frames += 1
        cv2.putText(frame, f"SEARCHING TARGET ID {target_id} ({missing_frames}/{MAX_MISSING_FRAMES})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if missing_frames > MAX_MISSING_FRAMES:
            print(f"[RESET] Target ID {target_id} lost. Unlocking.")
            target_id = None
            missing_frames = 0

    cv2.line(frame, (250, 0), (250, FRAME_HEIGHT), (100, 100, 100), 1)
    cv2.line(frame, (390, 0), (390, FRAME_HEIGHT), (100, 100, 100), 1)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Target Tracker & Mood - Pi 5", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("[MANUAL RESET] Target unlocked by user.")
        target_id = None
        missing_frames = 0

cap.release()
cv2.destroyAllWindows()