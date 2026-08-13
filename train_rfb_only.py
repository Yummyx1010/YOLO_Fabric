import sys, os
sys.path.append(os.path.dirname(__file__))

import ultralytics.nn.tasks as tasks
from rfb import RFB
tasks.RFB = RFB

from ultralytics import YOLO

def main():
    model = YOLO(r"D:\prog\PythonProg\YOLO_Fabric\models\yolov8n_rfb.yaml")

    model.train(
        data=r"D:\prog\PythonProg\YOLO_Fabric\datasetV2\data.yaml",
        imgsz=640,
        epochs=200,
        batch=16,
        device=0,
        pretrained=False,
        amp=True,
        workers=4,
        seed=0,

        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,

        project="runs_ablation",
        name="v8n_rfb_only_fromscratch_200epoch",
        exist_ok=True,
        cache=False,
        verbose=True,
    )

if __name__ == "__main__":
    main()
