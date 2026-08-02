python3 -m venv export_env
source export_env/bin/activate
pip install --upgrade pip
pip install ultralytics ncnn
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='ncnn', imgsz=320)"
python3 target_tracker.py
