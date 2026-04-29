#!/usr/bin/env python3
"""Sweep confidence thresholds for a YOLO-OBB model on a labeled OBB dataset.

Outputs JSON and CSV summaries so the operating point can be selected from real data.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ThresholdMetrics:
    threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds for YOLO-OBB inference")
    parser.add_argument("--model", required=True, help="Path to OBB model checkpoint")
    parser.add_argument(
        "--data",
        default="/Users/24900/_repos/yolo_train/dataset_obb_converted/data.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Dataset split")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    parser.add_argument("--iou", type=float, default=0.7, help="Prediction NMS IoU threshold")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for TP matching")
    parser.add_argument("--min-conf", type=float, default=0.05, help="Minimum confidence to evaluate")
    parser.add_argument("--max-conf", type=float, default=0.95, help="Maximum confidence to evaluate")
    parser.add_argument("--step", type=float, default=0.05, help="Threshold increment")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, mps, cuda:0")
    parser.add_argument(
        "--output-dir",
        default="/Users/24900/_repos/yolo_train/benchmarks/obb/threshold_sweep",
        help="Directory for JSON/CSV outputs",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_data_yaml(data_yaml: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = value.strip()
    return config


def resolve_split_dir(data_yaml: Path, split: str) -> tuple[Path, Path]:
    config = load_data_yaml(data_yaml)
    root = Path(config["path"]).expanduser().resolve() if "path" in config else data_yaml.parent.resolve()
    images_rel = config.get(split, f"images/{split}")
    images_dir = (root / images_rel).resolve()
    labels_dir = (root / "labels" / split).resolve()
    return images_dir, labels_dir


def find_images(images_dir: Path) -> list[Path]:
    files = []
    for ext in IMG_EXTENSIONS:
        files.extend(sorted(images_dir.glob(f"*{ext}")))
        files.extend(sorted(images_dir.glob(f"*{ext.upper()}")))
    return sorted(set(files))


def load_gt_xywhr(label_path: Path) -> torch.Tensor:
    if not label_path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)

    polys = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != 9:
            continue
        coords = [float(v) for v in toks[1:]]
        polys.append(
            [
                [coords[0], coords[1]],
                [coords[2], coords[3]],
                [coords[4], coords[5]],
                [coords[6], coords[7]],
            ]
        )

    if not polys:
        return torch.zeros((0, 5), dtype=torch.float32)
    poly_tensor = torch.tensor(polys, dtype=torch.float32)
    return xyxyxyxy2xywhr(poly_tensor)


def match_counts(pred_xywhr: torch.Tensor, pred_conf: torch.Tensor, gt_xywhr: torch.Tensor, conf_thresh: float, match_iou: float) -> tuple[int, int, int]:
    keep = pred_conf >= conf_thresh
    pred_xywhr = pred_xywhr[keep]
    pred_conf = pred_conf[keep]

    num_gt = int(gt_xywhr.shape[0])
    num_pred = int(pred_xywhr.shape[0])
    if num_gt == 0 and num_pred == 0:
        return 0, 0, 0
    if num_gt == 0:
        return 0, num_pred, 0
    if num_pred == 0:
        return 0, 0, num_gt

    # batch_probiou expects gt first, preds second and returns [num_gt, num_pred]
    iou_matrix = batch_probiou(gt_xywhr, pred_xywhr).cpu()
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    pred_order = torch.argsort(pred_conf, descending=True).tolist()
    for pred_idx in pred_order:
        best_gt = -1
        best_iou = match_iou
        for gt_idx in range(num_gt):
            if gt_idx in matched_gt:
                continue
            iou = float(iou_matrix[gt_idx, pred_idx])
            if iou >= best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0:
            matched_gt.add(best_gt)
            matched_pred.add(pred_idx)

    tp = len(matched_pred)
    fp = num_pred - tp
    fn = num_gt - len(matched_gt)
    return tp, fp, fn


def make_thresholds(min_conf: float, max_conf: float, step: float) -> list[float]:
    thresholds = []
    current = min_conf
    while current <= max_conf + 1e-9:
        thresholds.append(round(current, 6))
        current += step
    return thresholds


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    data_yaml = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    images_dir, labels_dir = resolve_split_dir(data_yaml, args.split)
    images = find_images(images_dir)
    thresholds = make_thresholds(args.min_conf, args.max_conf, args.step)

    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Data: {data_yaml}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Images: {len(images)}")
    print(f"[INFO] Device: {device}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(images_dir),
        imgsz=args.imgsz,
        conf=args.min_conf,
        iou=args.iou,
        device=device,
        verbose=False,
        max_det=300,
    )

    gt_by_stem: dict[str, torch.Tensor] = {}
    pred_by_stem: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for image_path, result in zip(images, results):
        gt_by_stem[image_path.stem] = load_gt_xywhr(labels_dir / f"{image_path.stem}.txt")
        if result.obb is None or len(result.obb.conf) == 0:
            pred_by_stem[image_path.stem] = (
                torch.zeros((0, 5), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
            )
            continue
        pred_xywhr = xyxyxyxy2xywhr(result.obb.xyxyxyxyn.cpu())
        pred_conf = result.obb.conf.cpu()
        pred_by_stem[image_path.stem] = (pred_xywhr, pred_conf)

    rows: list[ThresholdMetrics] = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for image_path in images:
            stem = image_path.stem
            gt_xywhr = gt_by_stem[stem]
            pred_xywhr, pred_conf = pred_by_stem[stem]
            image_tp, image_fp, image_fn = match_counts(pred_xywhr, pred_conf, gt_xywhr, threshold, args.match_iou)
            tp += image_tp
            fp += image_fp
            fn += image_fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        rows.append(ThresholdMetrics(threshold, tp, fp, fn, precision, recall, f1))

    best_f1 = max(rows, key=lambda row: row.f1)
    best_recall = max(rows, key=lambda row: row.recall)
    best_precision = max(rows, key=lambda row: row.precision)

    summary = {
        "model": str(model_path),
        "data_yaml": str(data_yaml),
        "split": args.split,
        "device": device,
        "match_iou": args.match_iou,
        "thresholds": [asdict(row) for row in rows],
        "best_f1": asdict(best_f1),
        "best_recall": asdict(best_recall),
        "best_precision": asdict(best_precision),
    }

    json_path = output_dir / f"threshold_sweep_{args.split}.json"
    csv_path = output_dir / f"threshold_sweep_{args.split}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
        for row in rows:
            writer.writerow([row.threshold, row.tp, row.fp, row.fn, row.precision, row.recall, row.f1])

    print(f"[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {csv_path}")
    print(f"[INFO] Best F1 threshold: {best_f1.threshold:.2f} (P={best_f1.precision:.4f}, R={best_f1.recall:.4f}, F1={best_f1.f1:.4f})")
    print(f"[INFO] Best recall threshold: {best_recall.threshold:.2f} (P={best_recall.precision:.4f}, R={best_recall.recall:.4f})")


if __name__ == "__main__":
    main()
