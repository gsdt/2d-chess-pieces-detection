# 2D Chess Board and Pieces Detection

Detect chess boards and pieces in 2D images using YOLO11n. Supports **multiple boards per image** (up to 12) and outputs FEN notation.

## Features

- Detects chess boards + 12 piece types (13 classes total)
- Supports 1-12 boards per image
- Works with various piece styles (chess.com, lichess, book diagrams)
- Model size: 5MB (PT) / 11MB (ONNX)
- Inference: ~30ms per image

## Quick Start

### Inference with pre-trained model

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("chess_image.png", conf=0.25, imgsz=1280)
```

Download trained models from the [Releases](https://github.com/Zai-Kun/2d-chess-pieces-detection/releases) page.

### ONNX inference (no PyTorch needed)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

def letterbox(img, size=1280):
    """Resize keeping aspect ratio + pad to square."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    canvas.paste(img.resize((nw, nh)), (px, py))
    return canvas, scale, px, py

session = ort.InferenceSession("best.onnx")
img = Image.open("chess_image.png").convert("RGB")
img_lb, scale, px, py = letterbox(img)
arr = np.expand_dims(np.transpose(np.array(img_lb, dtype=np.float32) / 255.0, (2,0,1)), 0)

outputs = session.run(None, {"images": arr})
# output shape: [1, 300, 6] = [x1, y1, x2, y2, confidence, class_id]
# coordinates are in 1280x1280 space, convert back:
# x_orig = (x - px) / scale
# y_orig = (y - py) / scale
```

> **Important:** Use letterbox preprocessing for non-square images. Direct resize will produce incorrect results.

## Classes

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | black_pawn | 6 | white_pawn |
| 1 | black_rook | 7 | white_rook |
| 2 | black_knight | 8 | white_knight |
| 3 | black_bishop | 9 | white_bishop |
| 4 | black_queen | 10 | white_queen |
| 5 | black_king | 11 | white_king |
| | | 12 | chess_board |

## Data Generation

### Single board (original)

```bash
python generate_datasets.py
```

Generates images with 1 board per image at 640x640.

### Multi-board (new)

```bash
python generate_multi_board.py
```

Generates images with 1-12 boards per image at 1280x1280. Config at top of file:

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | 1280 | Output image size |
| `BOARD_MIN_SIZE` | 120 | Minimum board size in pixels |
| `BOARD_MAX_SIZE` | 500 | Maximum board size in pixels |
| `MAX_BOARDS_PER_IMAGE` | 12 | Max boards per image |
| Train count | 5000 | Training images |
| Val count | 1000 | Validation images |

Both scripts use piece sets and board styles from `assets/` (30 boards, 60+ piece sets from chess.com and lichess).

## Training

### Local

```bash
python finetune.py
```

### Google Colab (recommended)

Upload `finetune_colab.ipynb` to [Google Colab](https://colab.research.google.com), select **GPU runtime**, and run all cells. Results are saved to Google Drive.

| GPU | Batch | Time (25 epochs) | Cost |
|---|---|---|---|
| T4 (free) | 8 | ~3 hours | $0 |
| A100 (Pro) | 20 | ~40 min | ~$1 |

### Resume training

```python
from ultralytics import YOLO
model = YOLO("runs/multi_board/weights/last.pt")
model.train(data="chess_detection_multi.yaml", epochs=50, resume=True)
```

## Project Structure

```
├── assets/
│   ├── boards/              # 30 board styles (.png)
│   └── pieces/              # 60+ piece sets
├── generate_datasets.py     # Single board data generation
├── generate_multi_board.py  # Multi-board data generation
├── random_fen_gen.py        # Random FEN position generator
├── finetune.py              # Local fine-tune script
├── finetune_colab.ipynb     # Colab notebook (all-in-one)
├── chess_detection.yaml     # Single board dataset config
└── 2d_chess_detection.ipynb # Original training notebook
```

## Results

Fine-tuned on 6000 multi-board images (25 epochs, A100):

| Metric | Value |
|---|---|
| mAP50 | 0.993 |
| mAP50-95 | 0.976 |
| Precision | 0.989 |
| Recall | 0.984 |

## Acknowledgments

Base model and assets from [Zai-Kun/2d-chess-pieces-detection](https://github.com/Zai-Kun/2d-chess-pieces-detection).
