import sys, os
sys.path.append(os.path.dirname(__file__))

import ultralytics.nn.tasks as tasks
from aspp import ASPP
tasks.ASPP = ASPP

from ultralytics import YOLO


def main():
    model = YOLO(r"D:\prog\PythonProg\YOLO_Fabric\models\yolov8n_aspp.yaml")

    model.train(
        data=r"D:\prog\PythonProg\YOLO_Fabric\datasetV2\data.yaml",
        imgsz=640,
        epochs=200,
        batch=16,
        device=0,

        pretrained=False,   # ✅保持消融公平
        amp=True,
        workers=4,
        seed=0,

        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,

        project="runs_ablation",
        name="v8n_aspp_only_fromscratch_200epoch",
        exist_ok=True,
        cache=False,
        verbose=True,
    )

if __name__ == "__main__":
    main()
