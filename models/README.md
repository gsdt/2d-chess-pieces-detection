# Trained Models

## v0.0.4-base

Original model from [Zai-Kun/2d-chess-pieces-detection](https://github.com/Zai-Kun/2d-chess-pieces-detection/releases/tag/v0.0.4).

- **Training:** Fine-tuned yolo11n.pt on single-board data, 45 epochs, imgsz=640
- **Data:** Single board per image, chess.com + lichess piece styles

| File | Size | Format |
|---|---|---|
| best.pt | 5.2MB | PyTorch |
| best.onnx | 10MB | ONNX (nms=True, imgsz=640) |

## v0.1.0-multi-board

Fine-tuned from v0.0.4-base on multi-board data with noise.

- **Training:** 25 epochs, imgsz=1280, batch=20, A100 GPU
- **Data:** 1-12 boards per image, 30 board styles, 60+ piece sets, background noise

| File | Size | Format |
|---|---|---|
| best.pt | 5.2MB | PyTorch |
| best.onnx | 11MB | ONNX (nms=True, imgsz=1280) |

| Metric | v0.0.4 | v0.1.0 |
|---|---|---|
| mAP50 | 0.993 | 0.993 |
| mAP50-95 | 0.958 | 0.976 |
| Precision | 0.987 | 0.989 |
| Recall | 0.980 | 0.984 |
| Multi-board | No | Yes (up to 12) |

## v0.2.0-noisy (next)

To be trained with additional noise: text-only negatives, image-only negatives, book-style layouts.
