======================================================================

🤖 COMPANION BOT — VISION \& TARGET LOCKING PROTOTYPE TEST

======================================================================



This repository contains the standalone vision testing module for the

Smart Autonomous Surveillance and Companion Robot\[cite: 1]. It tests real-time

human detection, motion tracking, visual re-identification (target 

locking), and steering offset computation using a webcam feed before 

deployment to the Raspberry Pi\[cite: 1].



\----------------------------------------------------------------------

📌 FEATURES TESTED

\----------------------------------------------------------------------

\* YOLOv8 Nano Detection: Fast person detection optimized for low-power 

&#x20; edge compute\[cite: 1].

\* ByteTrack Motion Tracking: Maintains object persistence across frame 

&#x20; sequences with minimal CPU overhead\[cite: 1].

\* HSV Visual Re-Identification (Target Locking): Generates a color 

&#x20; signature of the user's upper-body clothing so the robot can re-lock 

&#x20; onto the owner even if sight is temporarily lost\[cite: 1].

\* Stranger Rejection: Ignores other people walking through the frame\[cite: 1].

\* Steering Offset Computation: Calculates horizontal error (Delta X) 

&#x20; relative to the frame center to guide motor control\[cite: 1].



\----------------------------------------------------------------------

🛠️ PREREQUISITES \& SETUP

\----------------------------------------------------------------------



1\. Hardware Requirements:

&#x20;  - Laptop or PC running Windows, macOS, or Linux.

&#x20;  - Built-in laptop camera or external USB camera (e.g., Logitech Webcam)\[cite: 1].



2\. Environment Setup:

&#x20;  It is strongly recommended to run this project inside a Python 

&#x20;  virtual environment.



&#x20;  \[WARNING]: Do NOT run these scripts inside C:\\Windows\\System32. 

&#x20;  Create a dedicated project folder on your Desktop or User Directory.



&#x20;  Open your command line / terminal and run:



&#x20;  # Move to Desktop and create project folder

&#x20;  cd %USERPROFILE%\\Desktop

&#x20;  mkdir CompanionBot\_Test

&#x20;  cd CompanionBot\_Test



&#x20;  # Create and activate virtual environment

&#x20;  python -m venv venv



&#x20;  # On Windows:

&#x20;  venv\\Scripts\\activate



&#x20;  # On Linux/macOS:

&#x20;  source venv/bin/activate



3\. Dependency Installation:

&#x20;  Install the required Python packages:



&#x20;  pip install ultralytics opencv-python numpy



\----------------------------------------------------------------------

🚀 HOW TO RUN THE TEST SCRIPT

\----------------------------------------------------------------------



1\. Place `companion\_target\_locker.py` into your `CompanionBot\_Test` folder.

2\. Execute the script:



&#x20;  python companion\_target\_locker.py



3\. Note: On the first run, Ultralytics will automatically download 

&#x20;  the `yolov8n.pt` weight file (\~6 MB).



\----------------------------------------------------------------------

🎮 KEYBOARD CONTROLS

\----------------------------------------------------------------------

\[s] -> Save / Lock Target : Captures the visual profile of the person 

&#x20;      nearest to the camera center and locks onto them\[cite: 1].

\[r] -> Reset Lock         : Clears the saved profile memory, allowing 

&#x20;      you to lock onto a new target\[cite: 1].

\[q] -> Quit Application   : Safely stops the webcam capture feed and 

&#x20;      closes all display windows.



\----------------------------------------------------------------------

🧪 TESTING PROTOCOL \& CASE VERIFICATION

\----------------------------------------------------------------------



Test Case 1: Target Profile Lock

&#x20; 1. Stand in front of the camera and press 's'.

&#x20; 2. Expected Result: The status bar turns GREEN, a thick green 

&#x20;    bounding box surrounds you, and a label displays LOCKED TARGET #ID.



Test Case 2: Target Re-Identification (Frame Re-entry)

&#x20; 1. Lock onto yourself ('s'), then step completely out of view.

&#x20; 2. The status bar will turn RED ("SEARCHING FOR OWNER...").

&#x20; 3. Step back into the frame.

&#x20; 4. Expected Result: The system computes the HSV histogram match 

&#x20;    and automatically re-locks onto you with green bounding boxes\[cite: 1].



Test Case 3: Stranger Rejection

&#x20; 1. Lock onto yourself ('s').

&#x20; 2. Have a secondary person enter the camera frame alongside you.

&#x20; 3. Expected Result: The secondary person is enclosed in a thin RED 

&#x20;    box labeled IGNORED\[cite: 1].



Test Case 4: Steering Offset Output

&#x20; 1. Move left and right across the camera frame.

&#x20; 2. Observe the horizontal line and Steering Offset reading\[cite: 1].

&#x20; 3. Expected Result:

&#x20;    - Target on Left   : Offset reading is negative (e.g., -120px).

&#x20;    - Target Centered  : Offset reading is near zero (e.g., +5px).

&#x20;    - Target on Right  : Offset reading is positive (e.g., +140px).



\----------------------------------------------------------------------

🔍 TROUBLESHOOTING

\----------------------------------------------------------------------

\* Error: "ModuleNotFoundError: No module named 'cv2'"

&#x20; -> OpenCV not installed in active environment. Run: pip install opencv-python.

&#x20; 

\* Error: "Error: Could not open webcam"

&#x20; -> Incorrect camera index or device in use. Change cv2.VideoCapture(0) 

&#x20;    to 1 or 2 inside the script.



\* Target lock drops under dim lighting:

&#x20; -> Extreme lighting changes alter clothing HSV colors. Lower the 

&#x20;    similarity\_threshold in the script (e.g., from 0.50 to 0.40).



\----------------------------------------------------------------------

📋 NEXT STEPS FOR HARDWARE INTEGRATION

\----------------------------------------------------------------------

1\. Export YOLOv8 Nano model to NCNN or ONNX format for higher FPS 

&#x20;  on Raspberry Pi\[cite: 1].

2\. Replace print statements with direct UART/Serial write commands 

&#x20;  to send Steering Offset values to the ESP32 microcontroller\[cite: 1].

3\. Combine with MediaPipe Pose Estimation for fall detection integration\[cite: 1].

