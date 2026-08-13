from ultralytics import YOLO

def main():
    # ✅ from scratch：用 yaml 初始化
    model = YOLO("./models/yolov10n.yaml")

    model.train(
        data=r"D:\prog\PythonProg\YOLO_Fabric\datasetV2\data.yaml",
        imgsz=640,
        epochs=100,
        batch=16,
        device=0,

        # ✅核心协议
        pretrained=False,
        amp=True,

        # 默认增强：不手动覆盖增强参数
        workers=4,
        seed=0,

        project="runs_baseline",
        name="v10n_fromscratch",
        exist_ok=True,

        cache=False,
        verbose=True,
    )

if __name__ == "__main__":
    main()
