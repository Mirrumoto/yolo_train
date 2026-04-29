from __future__ import annotations

import argparse
import torch
from ultralytics import YOLO


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train latest YOLO nano OBB model")
    p.add_argument("--data", required=True, help="Dataset YAML path")
    p.add_argument("--model", default="yolo11n-obb.pt", help="Ultralytics OBB checkpoint")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="yolo11n_obb_firstpass_10k")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Data: {args.data}")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        amp=False,
        cache=False,
    )


if __name__ == "__main__":
    main()
