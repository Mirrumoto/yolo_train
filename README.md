# Synthetic MTG OBB Dataset Generator

Generates Ultralytics-compatible synthetic MTG card OBB datasets from Scryfall assets.

## Features

- `download-cards`: ingest Scryfall card fronts with caching and deduplication
- `prepare-backgrounds`: build procedural backgrounds and optional licensed texture manifest
- `generate`: synthesize scenes with card transforms, overlap control, effects, labels, metadata
- `validate`: geometry and dataset integrity checks
- `summarize`: QA summary across splits and augmentation/card-count distributions

## Quickstart

```bash
/Users/24900/_repos/.venv/bin/pip install -r requirements.txt

# 1) Download card images (front faces)
/Users/24900/_repos/.venv/bin/python -m synthetic_mtg_obb.cli download-cards \
  --out ./assets/cards --max-cards 500

# 2) Prepare backgrounds
/Users/24900/_repos/.venv/bin/python -m synthetic_mtg_obb.cli prepare-backgrounds \
  --out ./assets/backgrounds --procedural-count 300

# 3) Generate smoke dataset (100 images)
/Users/24900/_repos/.venv/bin/python -m synthetic_mtg_obb.cli generate \
  --cards-dir ./assets/cards \
  --backgrounds-dir ./assets/backgrounds \
  --out ./dataset \
  --num-images 100 \
  --seed 20260427 \
  --workers 1

# 4) Validate
/Users/24900/_repos/.venv/bin/python -m synthetic_mtg_obb.cli validate --dataset ./dataset

# 5) Summarize
/Users/24900/_repos/.venv/bin/python -m synthetic_mtg_obb.cli summarize --dataset ./dataset
```

## Output layout

```
dataset/
  dataset.yaml
  manifests/
    generation_manifest.jsonl
    generation_summary.json
    qa_summary.json
  images/
    train/
    val/
  labels/
    train/
    val/
  metadata/
    train/
    val/
```

## Label format

Each line in a label file:

`class x1 y1 x2 y2 x3 y3 x4 y4`

- `class`: integer (single class `0` for `card`)
- coordinates are normalized to `[0,1]`
- points are emitted clockwise

## Notes

- Target image format is `1024x1024` JPG quality `90` by default.
- Visibility gate default is `>= 90%` visible area.
- Split default is `90/10/0` (train/val/test).
- Reproducible with global seed and per-image deterministic local RNG.
- Human-behavior realism defaults include:
  - mixed camera mode,
  - tight within-image size consistency,
  - mostly-upright mixed orientation,
  - common sleeve simulation,
  - realistic dice/hand occluders,
  - mild motion blur.

See camera usage thresholds in `CAMERA_CAPTURE_GUIDELINES.md`.

## Frozen Application Model

Frozen production-candidate artifact:

- Model: `/Users/24900/_repos/yolo_train/frozen_models/yolo11n_obb_realworld_ft_20260428/model.pt`
- Manifest: `/Users/24900/_repos/yolo_train/frozen_models/yolo11n_obb_realworld_ft_20260428/model_manifest.json`
- Recommended runtime: `imgsz=1024`, `conf=0.80`, `iou=0.70`, `max_det=300`

Run app-style inference (annotated images + per-image JSON detections):

```bash
/Users/24900/_repos/.venv/bin/python /Users/24900/_repos/yolo_train/run_frozen_obb_inference.py \
  --input /ABS/PATH/TO/IMAGES_OR_IMAGE \
  # YOLO-OBB MTG Card Detection

  Training, inference, and synthetic dataset generation for Ultralytics YOLO-OBB oriented bounding box detection applied to Magic: The Gathering cards.

  ## Overview

  This project provides:
  - **Synthetic Dataset Generation**: Create large-scale training datasets from Scryfall card images with realistic augmentations
  - **Model Training**: Fine-tune YOLO11n-OBB on custom data
  - **Inference**: Single-image, batch, and real-time webcam detection
  - **Evaluation**: Model validation and performance benchmarking

  ## Quick Start

  ### 1. Setup

  ```bash
  git clone <repo-url>
  cd yolo_train

  # Create and activate virtual environment
  python3 -m venv venv
  source venv/bin/activate

  # Install dependencies
  pip install -r requirements.txt
  ```

  ### 2. Training

  Use an existing dataset (e.g., `dataset_firstpass_10000_human_behavior_v2/`) or generate one (see below).

  ```bash
  python train_obb.py \
    --data ./dataset_firstpass_10000_human_behavior_v2/dataset.yaml \
    --model yolo11n-obb.pt \
    --epochs 60 \
    --batch 8 \
    --imgsz 1024
  ```

  **Options:**
  - `--data`: Path to dataset YAML file (defines image paths, classes, splits)
  - `--model`: Base checkpoint (default: `yolo11n-obb.pt`)
  - `--epochs`: Number of training epochs
  - `--batch`: Batch size
  - `--imgsz`: Image size (1024 recommended)
  - `--resume`: Resume from last checkpoint

  Trained models are saved to `runs/train/`.

  ### 3. Inference

  Single image or directory:
  ```bash
  python run_frozen_obb_inference.py \
    --input ./input/image.jpg \
    --model runs/train/yolo11n_obb_firstpass_10k/weights/best.pt \
    --output ./output
  ```

  Real-time webcam:
  ```bash
  python live_webcam_obb_app.py \
    --camera-index 0 \
    --model runs/train/yolo11n_obb_firstpass_10k/weights/best.pt \
    --conf 0.6 \
    --imgsz 1024
  ```

  ## Generate Synthetic Dataset

  Use `synthetic_mtg_obb.cli` to build training data from Scryfall:

  ```bash
  # 1. Download card images (front faces)
  python -m synthetic_mtg_obb.cli download-cards \
    --out ./assets/cards \
    --max-cards 5000

  # 2. Prepare backgrounds
  python -m synthetic_mtg_obb.cli prepare-backgrounds \
    --out ./assets/backgrounds \
    --procedural-count 300

  # 3. Generate dataset (100k images)
  python -m synthetic_mtg_obb.cli generate \
    --cards-dir ./assets/cards \
    --backgrounds-dir ./assets/backgrounds \
    --out ./dataset_100k \
    --num-images 100000 \
    --seed 20260429

  # 4. Validate
  python -m synthetic_mtg_obb.cli validate --dataset ./dataset_100k

  # 5. Generate QA summary
  python -m synthetic_mtg_obb.cli summarize --dataset ./dataset_100k
  ```

  **Generated Structure:**
  ```
  dataset_100k/
  ├── dataset.yaml           # Config for training
  ├── images/
  │   ├── train/             # 90% of images
  │   └── val/               # 10% of images
  ├── labels/
  │   ├── train/             # OBB labels (normalized)
  │   └── val/
  └── manifests/
      ├── generation_manifest.jsonl
      ├── generation_summary.json
      └── qa_summary.json
  ```

  ## Label Format

  Each label file contains one line per detected card:
  ```
  class x1 y1 x2 y2 x3 y3 x4 y4
  ```

  - `class`: `0` for card (single class)
  - Coordinates: normalized to `[0, 1]`
  - Points: ordered clockwise
  - Example: `0 0.1 0.2 0.9 0.15 0.95 0.8 0.25 0.85`

  ## Dataset Structure

  Your dataset YAML should follow Ultralytics format:

  ```yaml
  path: /path/to/dataset
  train: images/train
  val: images/val
  test: images/test

  nc: 1
  names:
    0: card
  ```

  ## Files

  | File | Purpose |
  |------|---------|
  | `train_obb.py` | Main training script |
  | `run_frozen_obb_inference.py` | Batch/single-image inference |
  | `live_webcam_obb_app.py` | Real-time webcam detection |
  | `benchmark_obb_pipeline.py` | Performance benchmarking |
  | `threshold_sweep_obb.py` | Confidence threshold analysis |
  | `synthetic_mtg_obb/` | Dataset generation CLI |

  ## Device Support

  - **GPU**: Automatically uses CUDA (if available)
  - **Apple Silicon**: Uses MPS (Metal Performance Shaders)
  - **CPU**: Falls back to CPU (slower)

  ## Configuration

  Key hyperparameters (in `train_obb.py`):
  - Image size: `1024x1024` (standard)
  - Model: `yolo11n-obb.pt` (nano, fast)
  - Batch size: 8 (adjust based on VRAM)
  - Learning rate: Ultralytics default (0.01)

  Adjust in `train_obb.py` or via CLI args.

  ## Model Management

  - **Checkpoints**: Saved in `runs/train/<run-name>/weights/`
    - `last.pt`: Latest epoch
    - `best.pt`: Highest validation accuracy
  - **Model size**: ~40-50 MB for nano model
  - **DO NOT commit models to Git** - store locally or use cloud storage (S3, HuggingFace Hub, etc.)

  ## Troubleshooting

  **OOM (Out of Memory)**
  - Reduce `--batch` size (e.g., 4 or 2)
  - Reduce `--imgsz` (e.g., 512)
  - Use smaller model (yolo11n vs yolo11m)

  **Slow training**
  - Check GPU is being used: look for GPU utilization in output
  - Increase workers: `--workers 8` (if CPU has cores)

  **Camera not detected**
  - Check camera index: `--camera-index 0` (try 1, 2 if needed)
  - Test with OpenCV: `python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"`

  ## References

  - [Ultralytics YOLO-OBB Docs](https://docs.ultralytics.com/tasks/obb/)
  - [Scryfall API](https://scryfall.com/docs/api)
  - [MTG Card Detection Guidelines](./CAMERA_CAPTURE_GUIDELINES.md)

  ## License

  This project is for research and hobby use. MTG card images are property of Wizards of the Coast.
