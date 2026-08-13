from ultralytics import YOLO

def main():
    # ✅ 用项目内的 yolov8n.yaml（建议从 site-packages 复制一份到 ./models/）
    model = YOLO("./models/yolov8n.yaml")

    model.train(
        data=r"D:\prog\PythonProg\YOLO_Fabric\datasetV2\data.yaml",  # ✅和v5n一致：同一份data.yaml
        imgsz=640,
        epochs=200,
        batch=16,
        device=0,

        # ✅核心：from scratch + AMP
        pretrained=False,
        amp=True,

        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,

        # 笔记本建议别太大
        workers=4,
        seed=0,
        

        # 统一输出目录
        project="runs_baseline",
        name="v8n_fromscratch_200epoch_v2",
        exist_ok=True,

        # ✅默认增强：不要显式设置 mosaic/mixup/hsv 等，让 Ultralytics 自己用默认
        cache=False,
        verbose=True,
    )

if __name__ == "__main__":
    main()
