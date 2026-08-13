
## Dataset

- **Name**: Fabric Defect Dataset 
- **Classes**: 6 defect categories
- **Image size**: 640 × 640


## Quick Start

### 1. Environment

```bash
git clone https://github.com/Yummyx1010/YOLO_Fabric.git
cd YOLO_Fabric
pip install ultralytics torch torchvision
```

### 2. Train the Proposed Model (YOLOv8n + ASPP + RFB)

```bash
yolo detect train \
    model=models/yolov8n_rfb_aspp.yaml \
    data=datasetV2/data.yaml \
    epochs=200 \
    imgsz=640 \
    batch=16 \
    device=0 \
    workers=4 \
    optimizer=SGD \
    project=runs_ablation \
    name=v8n_rfb_aspp_fromscratch_200epoch
```

### 3. Inference

```bash
yolo detect predict model=path/to/best.pt source=path/to/images imgsz=640
```

## Experiments

All experiments are trained **from scratch** for 200 epochs under the same hyperparameters:

| Configuration |  mAP@50 | mAP@50-95 |
|---------------|--------|-----------|
| YOLOv8n (baseline) | 0.8759 | 0.4820 |
| YOLOv8n + ASPP only |  0.8850 | 0.4388 |
| YOLOv8n + RFB only | 0.7544 | 0.3704 |
| **YOLOv8n + ASPP + RFB (Ours)** | **0.9402** | **0.4805** |

**Training settings**:

| Item | Value |
|------|-------|
| Epochs | 200 |
| Batch size | 16 |
| Image size | 640 |
| Optimizer | SGD |
| Patience | 100 |
| Close mosaic | last 10 epochs |
| AMP | enabled |
| Pretrained | False (from scratch) |

### Ablation Insights

- **ASPP only** brings a small gain in mAP@50 (+0.9%) but slightly drops mAP@50-95.
- **RFB only** underperforms when used alone, indicating that RFB benefits from the richer multi-scale context provided by ASPP.
- **ASPP + RFB (Ours)** achieves the best mAP@50 = **94.02%**, a **+6.4%** improvement over the YOLOv8n baseline, demonstrating that the two modules are complementary.

## Result Files

Each experiment folder under `runs_ablation/` and `runs_baseline/` contains:

- `args.yaml` — full training configuration
- `results.csv` — per-epoch metrics
- `results.png` — training curves
- `confusion_matrix.png` — confusion matrix
- `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png` — PR/F1/P/R curves
- `train_batch*.jpg`, `val_batch*_pred.jpg` — visualization samples


##  License

This project is intended for academic and research use only.

##  Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Original ASPP from DeepLabv3
- Original RFB from "Receptive Field Block Net"
