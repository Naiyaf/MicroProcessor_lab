import cv2
import numpy as np
from ultralytics import YOLO

def get_upper_body_histogram(frame, box):
    """
    Extracts an HSV color histogram from the upper half (torso/clothing) 
    of a detected person bounding box.
    """
    x1, y1, x2, y2 = box
    h = y2 - y1
    upper_y2 = y1 + int(h * 0.5)  # Isolate torso/shirt area
    
    # Clamp coordinates to frame boundaries
    height, width, _ = frame.shape
    x1_clamped = max(0, min(width - 1, x1))
    x2_clamped = max(0, min(width - 1, x2))
    y1_clamped = max(0, min(height - 1, y1))
    upper_y2_clamped = max(0, min(height - 1, upper_y2))

    crop = frame[y1_clamped:upper_y2_clamped, x1_clamped:x2_clamped]
    if crop.size == 0:
        return None

    # Convert RGB crop to HSV space
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Compute 2D Hue-Saturation Histogram
    hist = cv2.calcHist([hsv_crop], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def compare_histograms(hist1, hist2):
    """
    Calculates correlation similarity between two visual profiles (0.0 to 1.0).
    """
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def main():
    # Load YOLOv8 Nano model
    model = YOLO('yolov8n.pt')

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    locked_signature = None
    locked_target_id = None
    similarity_threshold = 0.50  # Adjust between 0.40 and 0.70 depending on lighting

    print("\n=======================================================")
    print("      COMPANION BOT - TARGET LOCKING TEST RIG")
    print("=======================================================")
    print(" -> Stand in front of the camera and press 's' to LOCK.")
    print(" -> Press 'r' to RESET/FORGET target.")
    print(" -> Press 'q' to QUIT.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w, _ = frame.shape
        frame_center_x = frame_w // 2

        # Run YOLOv8 Nano + ByteTrack on person class only (class 0)
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False
        )

        current_candidates = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.int().cpu().tolist()

            best_match_id = None
            highest_sim = 0.0

            # Step 1: Extract features and check visual match against saved profile
            for box, track_id in zip(boxes, track_ids):
                hist = get_upper_body_histogram(frame, box)
                
                if hist is not None:
                    current_candidates.append({
                        'id': track_id,
                        'box': box,
                        'hist': hist
                    })

                    if locked_signature is not None:
                        sim = compare_histograms(locked_signature, hist)
                        if sim > highest_sim and sim >= similarity_threshold:
                            highest_sim = sim
                            best_match_id = track_id

            # Step 2: Update target ID based on visual re-identification match
            if locked_signature is not None:
                locked_target_id = best_match_id

            # Step 3: Draw bounding boxes and steering overlays
            for candidate in current_candidates:
                x1, y1, x2, y2 = candidate['box']
                t_id = candidate['id']
                
                if t_id == locked_target_id:
                    # Calculate center point and horizontal steering offset
                    target_center_x = (x1 + x2) // 2
                    x_offset = target_center_x - frame_center_x

                    # Draw GREEN box for owner
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    label = f"LOCKED TARGET #{t_id} (Match: {highest_sim * 100:.0f}%)"
                    offset_label = f"Steering Offset: {x_offset:+d}px"
                    
                    cv2.putText(frame, label, (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cv2.putText(frame, offset_label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
                    
                    # Draw target center dot and offset line
                    target_center_y = (y1 + y2) // 2
                    cv2.circle(frame, (target_center_x, target_center_y), 6, (0, 255, 0), -1)
                    cv2.line(frame, (frame_center_x, target_center_y), 
                             (target_center_x, target_center_y), (255, 255, 0), 2)

                else:
                    # Draw RED box for non-targets / strangers
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, f"IGNORED #{t_id}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Draw frame center vertical alignment line
        cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame_h), (255, 255, 255), 1)

        # On-screen HUD Status Bar
        if locked_signature is None:
            status_text = "STATUS: UNLOCKED (Center yourself & press 's')"
            status_color = (0, 165, 255)  # Orange
        elif locked_target_id is None:
            status_text = "STATUS: SEARCHING FOR OWNER..."
            status_color = (0, 0, 255)  # Red
        else:
            status_text = f"STATUS: TRACKING OWNER #{locked_target_id}"
            status_color = (0, 255, 0)  # Green

        cv2.putText(frame, status_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.imshow("Companion Bot - Target Locking Test", frame)

        # Handle keyboard interactions
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if current_candidates:
                # Lock onto the person nearest to the camera frame center
                best_cand = min(
                    current_candidates,
                    key=lambda c: abs(((c['box'][0] + c['box'][2]) // 2) - frame_center_x)
                )
                locked_signature = best_cand['hist']
                locked_target_id = best_cand['id']
                print(f"\n[LOCKED] Saved visual profile for Person #{locked_target_id}!")
            else:
                print("\n[WARNING] No person detected in frame to lock onto!")
        elif key == ord('r'):
            locked_signature = None
            locked_target_id = None
            print("\n[RESET] Target profile cleared.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()