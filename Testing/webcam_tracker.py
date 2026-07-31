import cv2
from ultralytics import YOLO

def main():
    # Load the pre-trained YOLOv8 Nano model
    model = YOLO('yolov8n.pt') 

    # Open the default webcam (usually index 0)
    # If you have an external Logitech camera plugged in, it might be index 1
    cap = cv2.VideoCapture(0)
    
    # Set webcam resolution (optional, but good for performance testing)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit the webcam feed.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Run YOLOv8 tracking with ByteTrack
        # persist=True keeps the tracking IDs consistent across frames
        # tracker="bytetrack.yaml" specifies the tracking algorithm
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0]) # classes=[0] tracks only "person"

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 + ByteTrack Webcam Test", annotated_frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()