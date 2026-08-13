from ultralytics import YOLO

def main():
    # ✅ 用项目内的 yolov8n.yaml（建议从 site-packages 复制一份到 ./models/）
    model = YOLO("./models/yolov5n.yaml")

    model.train(
        data=r"D:\prog\PythonProg\YOLO_Fabric\datasetV2\data.yaml",  # ✅和v5n一致：同一份data.yaml
        imgsz=640,
        epochs=100,
        batch=16,
        device=0,

        # ✅核心：from scratch + AMP
        pretrained=False,
        amp=True,

        # 笔记本建议别太大
        workers=4,
        seed=0,
        

        # 统一输出目录
        project="runs_baseline",
        name="v5n_fromscratch",
        exist_ok=True,

        # ✅默认增强：不要显式设置 mosaic/mixup/hsv 等，让 Ultralytics 自己用默认
        cache=False,
        verbose=True,
    )

if __name__ == "__main__":
    main()
