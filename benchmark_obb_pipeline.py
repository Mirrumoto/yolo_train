#!/usr/bin/env python3
"""Benchmark a trained YOLO-OBB model against a real-world OBB dataset.

Pipeline stages:
1) Audit dataset structure and OBB label validity.
2) Run Ultralytics validation on requested split.
3) Export machine-readable reports for tracking and comparisons.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class AuditSummary:
    images_train: int = 0
    labels_train: int = 0
    images_val: int = 0
    labels_val: int = 0
    images_test: int = 0
    labels_test: int = 0
    total_images: int = 0
    total_label_files: int = 0
    bad_field_count: int = 0
    bad_class_count: int = 0
    bad_coord_range_count: int = 0
    degenerate_polygon_count: int = 0
    labels_missing_image_count: int = 0
    images_missing_label_count: int = 0

    @property
    def is_valid(self) -> bool:
        return (
            self.bad_field_count == 0
            and self.bad_class_count == 0
            and self.bad_coord_range_count == 0
            and self.degenerate_polygon_count == 0
            and self.labels_missing_image_count == 0
            and self.images_missing_label_count == 0
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a YOLO-OBB checkpoint on a real OBB dataset")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model checkpoint (.pt), e.g. best.pt",
    )
    parser.add_argument(
        "--data",
        default="/Users/24900/_repos/yolo_train/dataset_obb_converted/data.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
        help="Dataset split to benchmark",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Validation image size")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size")
    parser.add_argument("--workers", type=int, default=4, help="Validation dataloader workers")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, mps, or cuda:0",
    )
    parser.add_argument(
        "--project",
        default="/Users/24900/_repos/yolo_train/benchmarks/obb",
        help="Benchmark output root directory",
    )
    parser.add_argument("--name", default="realworld_obb_benchmark", help="Run name under project")
    parser.add_argument(
        "--strict-audit",
        action="store_true",
        help="Fail fast if dataset audit finds structural/format issues",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += (x1 * y2) - (y1 * x2)
    return abs(area) * 0.5


def audit_dataset(data_yaml: Path) -> AuditSummary:
    root = data_yaml.parent
    summary = AuditSummary()

    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        image_files = sorted([p for p in img_dir.glob("*") if p.is_file() and is_image_file(p)]) if img_dir.exists() else []
        label_files = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        setattr(summary, f"images_{split}", len(image_files))
        setattr(summary, f"labels_{split}", len(label_files))

        # label -> image pairing
        for label in label_files:
            stem = label.stem
            matched = any((img_dir / f"{stem}{ext}").exists() for ext in IMG_EXTENSIONS)
            if not matched:
                summary.labels_missing_image_count += 1

        # image -> label pairing
        for image in image_files:
            if not (lbl_dir / f"{image.stem}.txt").exists():
                summary.images_missing_label_count += 1

        # line-level OBB validation
        for label in label_files:
            for raw in label.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                toks = line.split()
                if len(toks) != 9:
                    summary.bad_field_count += 1
                    continue

                cls_tok = toks[0]
                if not cls_tok.isdigit() or int(cls_tok) != 0:
                    summary.bad_class_count += 1

                try:
                    coords = [float(t) for t in toks[1:]]
                except ValueError:
                    summary.bad_coord_range_count += 1
                    continue

                if any(v < 0.0 or v > 1.0 for v in coords):
                    summary.bad_coord_range_count += 1
                    continue

                pts = [(coords[0], coords[1]), (coords[2], coords[3]), (coords[4], coords[5]), (coords[6], coords[7])]
                if polygon_area(pts) <= 1e-6:
                    summary.degenerate_polygon_count += 1

    summary.total_images = summary.images_train + summary.images_val + summary.images_test
    summary.total_label_files = summary.labels_train + summary.labels_val + summary.labels_test
    return summary


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "__dict__"):
        return to_jsonable(vars(obj))
    return str(obj)


def extract_metrics(metrics_obj: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    # Preferred source in Ultralytics.
    if hasattr(metrics_obj, "results_dict"):
        metrics.update(to_jsonable(metrics_obj.results_dict))

    # Keep speed and save dir explicit for convenience.
    if hasattr(metrics_obj, "speed"):
        metrics["speed_ms"] = to_jsonable(metrics_obj.speed)
    if hasattr(metrics_obj, "save_dir"):
        metrics["save_dir"] = str(metrics_obj.save_dir)

    return metrics


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    data_yaml = Path(args.data).expanduser().resolve()
    output_root = Path(args.project).expanduser().resolve() / args.name
    output_root.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    device = get_device(args.device)

    print("=" * 72)
    print("YOLO OBB Benchmark Pipeline")
    print("=" * 72)
    print(f"[INFO] Model:  {model_path}")
    print(f"[INFO] Data:   {data_yaml}")
    print(f"[INFO] Split:  {args.split}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output: {output_root}")

    print("\n[1/3] Auditing dataset...")
    audit = audit_dataset(data_yaml)
    audit_report = asdict(audit)
    audit_report["is_valid"] = audit.is_valid
    (output_root / "audit_summary.json").write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    print("[1/3] Done")
    print(f"        valid={audit.is_valid} images={audit.total_images} label_files={audit.total_label_files}")

    if args.strict_audit and not audit.is_valid:
        raise RuntimeError("Dataset audit failed. See audit_summary.json for details.")

    print("\n[2/3] Running OBB validation...")
    model = YOLO(str(model_path))
    val_metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        device=device,
        task="obb",
        plots=True,
        save_json=True,
        project=str(output_root),
        name="val",
    )
    print("[2/3] Done")

    print("\n[3/3] Writing reports...")
    extracted = extract_metrics(val_metrics)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "data_yaml": str(data_yaml),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "device": device,
        "audit": audit_report,
        "metrics": extracted,
    }

    (output_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "metrics_results_dict.json").write_text(
        json.dumps(to_jsonable(getattr(val_metrics, "results_dict", {})), indent=2),
        encoding="utf-8",
    )
    print("[3/3] Done")

    print("\nBenchmark complete")
    print(f"Summary: {output_root / 'benchmark_summary.json'}")
    print(f"Audit:   {output_root / 'audit_summary.json'}")


if __name__ == "__main__":
    main()
