# YOLO-OBB MTG Card Detection

Train and run YOLO-OBB for MTG card detection, with optional synthetic dataset generation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train_obb.py \
  --data ./dataset_synthetic/dataset.yaml \
  --model yolo11n-obb.pt \
  --epochs 60 \
  --batch 8 \
  --imgsz 1024
```

Useful flags: `--resume`, `--workers`, `--name`, `--project`.

## Inference

```bash
python run_frozen_obb_inference.py \
  --input ./input \
  --model runs/train/<run-name>/weights/best.pt \
  --output ./output
```

```bash
python live_webcam_obb_app.py \
  --camera-index 0 \
  --model runs/train/<run-name>/weights/best.pt \
  --conf 0.6 \
  --imgsz 1024
```

## Synthetic Data (Optional)

```bash
python -m synthetic_mtg_obb.cli download-cards --out ./assets/cards --max-cards 5000
python -m synthetic_mtg_obb.cli prepare-backgrounds --out ./assets/backgrounds --procedural-count 300
python -m synthetic_mtg_obb.cli generate --cards-dir ./assets/cards --backgrounds-dir ./assets/backgrounds --out ./dataset --num-images 100000 --seed 20260429
python -m synthetic_mtg_obb.cli validate --dataset ./dataset
python -m synthetic_mtg_obb.cli summarize --dataset ./dataset
```

OBB labels use:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```
