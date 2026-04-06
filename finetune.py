"""Fine-tune YOLO11n chess detection model with multi-board data.

Usage:
    # Step 1: Generate data
    python generate_multi_board.py

    # Step 2: Fine-tune
    python finetune.py
"""

from ultralytics import YOLO

# Load the existing trained model
model = YOLO("best.pt")

# Fine-tune with multi-board data at 1280x1280
model.train(
    data="chess_detection_multi.yaml",
    epochs=15,
    imgsz=640,       # train at 640, can infer at 1280 later
    batch=32,        # large batch = faster
    device="mps",    # Apple Silicon GPU
    lr0=0.0001,      # low lr to preserve existing knowledge
    optimizer="AdamW",
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    scale=0.5,
    degrees=5,
    project="runs/finetune",
    name="multi_board_640",
    exist_ok=True,
)

# Export to ONNX
best = YOLO("runs/finetune/multi_board_640/weights/best.pt")
best.export(format="onnx", imgsz=640, nms=True)

print("\nDone! Model saved to:")
print("  PT:   runs/finetune/multi_board_640/weights/best.pt")
print("  ONNX: runs/finetune/multi_board_640/weights/best.onnx")
