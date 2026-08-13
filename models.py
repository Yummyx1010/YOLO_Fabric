from ultralytics import YOLO
from pathlib import Path
weights = Path("./models/yolov8n.pt")
m = YOLO(str(weights))
print(m.model.yaml)
