#!/usr/bin/env python3
"""Run inference with the frozen OBB model and export app-friendly outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


DEFAULT_MODEL = "/Users/24900/_repos/yolo_train/frozen_models/yolo11n_obb_realworld_ft_20260428/model.pt"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen YOLO-OBB model for app testing")
    parser.add_argument("--input", required=True, help="Image file or folder of images")
    parser.add_argument("--output", required=True, help="Output folder for images and JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to frozen model artifact")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference size")
    parser.add_argument("--conf", type=float, default=0.8, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", default="mps", help="Device: cpu, mps, cuda:0, ...")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    return parser.parse_args()


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image file extension: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    files = [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise ValueError(f"No images found in: {input_path}")
    return files


def as_float_list(arr: np.ndarray) -> list[float]:
    return [float(x) for x in arr.tolist()]


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_images = output_root / "images"
    output_json = output_root / "json"
    output_images.mkdir(parents=True, exist_ok=True)
    output_json.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_path)
    source = [str(p) for p in image_paths]

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
    )

    summary: list[dict] = []

    for image_path, result in zip(image_paths, results):
        detections = []
        if result.obb is not None and len(result.obb.conf) > 0:
            xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
            xyxyxyxyn = result.obb.xyxyxyxyn.cpu().numpy()
            conf = result.obb.conf.cpu().numpy()
            cls = result.obb.cls.cpu().numpy()

            for idx in range(len(conf)):
                detections.append(
                    {
                        "class_id": int(cls[idx]),
                        "confidence": float(conf[idx]),
                        "poly_xyxyxyxy": as_float_list(xyxyxyxy[idx].reshape(-1)),
                        "poly_xyxyxyxy_norm": as_float_list(xyxyxyxyn[idx].reshape(-1)),
                    }
                )

        plotted = result.plot()
        plotted_rgb = plotted[..., ::-1]
        annotated_path = output_images / f"{image_path.stem}_annotated.jpg"
        Image.fromarray(plotted_rgb).save(annotated_path, quality=90)

        per_image = {
            "input_image": str(image_path),
            "annotated_image": str(annotated_path),
            "num_detections": len(detections),
            "detections": detections,
        }
        summary.append(per_image)

        with (output_json / f"{image_path.stem}.json").open("w", encoding="utf-8") as f:
            json.dump(per_image, f, indent=2)

    run_summary = {
        "model": str(model_path),
        "input": str(input_path),
        "output": str(output_root),
        "runtime": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "max_det": args.max_det,
        },
        "images_processed": len(image_paths),
        "results": summary,
    }

    with (output_root / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(f"[INFO] Processed {len(image_paths)} images")
    print(f"[INFO] Outputs written to: {output_root}")


if __name__ == "__main__":
    main()
