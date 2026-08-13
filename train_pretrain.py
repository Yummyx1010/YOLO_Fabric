from ultralytics import YOLO
from pathlib import Path

def main():
    weights = Path("./models/yolov8n.pt")
    model = YOLO(str(weights))

    model.train(
        data=r"D:\prog\PythonProg\YOLO_Fabric\datasetV2\data.yaml",
        imgsz=640,
        epochs=100,
        batch=16,
        device=0,

        amp=True,
        workers=4,
        

        project="runs_reference",   # ⬅️ 不和 v8 baseline 混
        name="v8n_pretrained2",
        exist_ok=False,

        cache=False,
        verbose=True,
    )

if __name__ == "__main__":
    main()
