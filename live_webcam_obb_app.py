#!/usr/bin/env python3
"""Real-time webcam app for frozen YOLO-OBB card detection."""

from __future__ import annotations

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL = "/Users/24900/_repos/yolo_train/frozen_models/yolo11n_obb_realworld_ft_20260428/model.pt"
WINDOW_NAME = "MTG Card Detection (Live)"


@dataclass
class SavedCard:
    hash_bits: np.ndarray
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class CropCandidate:
    hash_bits: np.ndarray
    cx: float
    cy: float
    w: float
    h: float
    best_crop: np.ndarray
    best_conf: float
    best_sharpness: float
    best_score: float
    first_frame: int
    last_frame: int
    seen_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live webcam card detection with frozen YOLO-OBB model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to frozen model artifact")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--imgsz", type=int, default=768, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (use lower values for diagonal cards)")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", default="mps", help="Inference device: mps, cpu, cuda:0, ...")
    parser.add_argument("--max-det", type=int, default=120, help="Maximum detections per frame")
    parser.add_argument("--infer-every", type=int, default=2, help="Run model every N frames (>=1)")
    parser.add_argument(
        "--inference-scale",
        type=float,
        default=0.75,
        help="Scale factor for inference frame (0.25-1.0). Lower is faster.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested camera frame width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera frame height")
    parser.add_argument(
        "--rotate",
        default="none",
        choices=("none", "cw", "ccw", "180"),
        help="Rotate frames before inference/display if camera reports portrait orientation",
    )
    parser.add_argument("--line-thickness", type=int, default=2, help="Bounding polygon line thickness")
    parser.add_argument("--save-video", action="store_true", help="Save annotated live stream to disk")
    parser.add_argument(
        "--output-dir",
        default="/Users/24900/_repos/yolo_train/benchmarks/obb/live_recordings",
        help="Output directory for recorded videos",
    )
    parser.add_argument("--output-name", default="", help="Optional output filename (without extension)")
    parser.add_argument("--output-fps", type=float, default=30.0, help="Recorded video FPS")
    parser.add_argument("--record-every", type=int, default=1, help="Write every Nth frame to recorded video (>=1)")
    parser.add_argument("--save-crops", action="store_true", help="Save detected card crops once per unique card")
    parser.add_argument(
        "--crops-dir",
        default="/Users/24900/_repos/yolo_train/benchmarks/obb/live_crops",
        help="Output directory for saved card crops",
    )
    parser.add_argument("--crop-min-conf", type=float, default=0.7, help="Minimum confidence to save a crop")
    parser.add_argument("--crop-output-height", type=int, default=1024, help="Output height for rectified card crops")
    parser.add_argument("--crop-jpeg-quality", type=int, default=98, help="JPEG quality for saved crops (1-100)")
    parser.add_argument("--crop-margin", type=float, default=0.03, help="Relative margin added around detected card quad")
    parser.add_argument(
        "--crop-enhance",
        default="sharpen",
        choices=("none", "sharpen"),
        help="Optional postprocess for saved crops",
    )
    parser.add_argument(
        "--crop-hash-threshold",
        type=int,
        default=4,
        help="Hamming distance threshold for deduping visually similar crops",
    )
    parser.add_argument("--crop-dedupe-iou", type=float, default=0.4, help="Spatial IoU threshold for duplicate suppression")
    parser.add_argument("--crop-track-iou", type=float, default=0.35, help="IoU threshold to associate detections to the same crop candidate")
    parser.add_argument("--crop-settle-frames", type=int, default=12, help="Frames to wait after last sighting before selecting best crop")
    parser.add_argument("--crop-min-observations", type=int, default=2, help="Minimum sightings before candidate can be saved")
    parser.add_argument("--crop-min-sharpness", type=float, default=60.0, help="Minimum Laplacian variance required to save crop")
    parser.add_argument("--crop-debug-every", type=int, default=60, help="Print crop pipeline debug stats every N frames (0 disables)")
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open webcam. Check camera permissions for Terminal/VS Code and try again."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def apply_rotation(frame: np.ndarray, rotate: str) -> np.ndarray:
    if rotate == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def resolve_model_path(model_arg: str) -> Path:
    requested = Path(model_arg).expanduser()
    if not requested.is_absolute():
        requested = (Path.cwd() / requested).resolve()
    else:
        requested = requested.resolve()

    if requested.exists():
        return requested

    # Friendly fallback for common typo: passing "model.pt" from repo root.
    default_path = Path(DEFAULT_MODEL).expanduser().resolve()
    if default_path.exists():
        print(f"[WARN] Model not found: {requested}")
        print(f"[WARN] Falling back to frozen model: {default_path}")
        return default_path

    raise FileNotFoundError(
        "Model not found and fallback unavailable. "
        f"Requested: {requested} | Fallback: {default_path}"
    )


def make_video_writer(args: argparse.Namespace) -> tuple[cv2.VideoWriter | None, Path | None]:
    if not args.save_video:
        return None, None

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.output_name.strip() if args.output_name else datetime.now().strftime("live_obb_%Y%m%d_%H%M%S")
    output_path = output_dir / f"{base_name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.output_fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for: {output_path}")
    return writer, output_path


def draw_detections(frame: np.ndarray, result, line_thickness: int) -> int:
    det_count = 0
    if result.obb is None or len(result.obb.conf) == 0:
        return det_count

    polys = result.obb.xyxyxyxy.cpu().numpy()
    confs = result.obb.conf.cpu().numpy()

    for poly, conf in zip(polys, confs):
        det_count += 1
        pts = poly.reshape(4, 2).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(64, 255, 64), thickness=line_thickness)

        anchor = pts[0]
        label = f"card {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (int(anchor[0]), int(anchor[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (64, 255, 64),
            2,
            cv2.LINE_AA,
        )

    return det_count


def draw_cached_detections(frame: np.ndarray, polys: np.ndarray, confs: np.ndarray, line_thickness: int) -> int:
    det_count = 0
    if polys.size == 0 or confs.size == 0:
        return det_count

    for poly, conf in zip(polys, confs):
        det_count += 1
        pts = poly.reshape(4, 2).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(64, 255, 64), thickness=line_thickness)

        anchor = pts[0]
        label = f"card {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (int(anchor[0]), int(anchor[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (64, 255, 64),
            2,
            cv2.LINE_AA,
        )

    return det_count


def extract_polys_confs(result, scale_x: float, scale_y: float) -> tuple[np.ndarray, np.ndarray]:
    if result.obb is None or len(result.obb.conf) == 0:
        return np.empty((0, 4, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    polys = result.obb.xyxyxyxy.cpu().numpy().astype(np.float32)
    polys[:, :, 0] *= scale_x
    polys[:, :, 1] *= scale_y
    confs = result.obb.conf.cpu().numpy().astype(np.float32)
    return polys, confs


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    # Rotate so the first point is approximately top-left for stable warping.
    start_idx = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start_idx, axis=0).astype(np.float32)


def expand_quad(rect: np.ndarray, margin: float) -> np.ndarray:
    if margin <= 0:
        return rect
    c = rect.mean(axis=0, keepdims=True)
    return c + (rect - c) * (1.0 + margin)


def crop_from_polygon(frame: np.ndarray, poly: np.ndarray, output_height: int, margin: float) -> np.ndarray | None:
    pts = poly.reshape(4, 2).astype(np.float32)
    rect = _order_quad_points(pts)
    rect = expand_quad(rect, margin)

    width_top = np.linalg.norm(rect[1] - rect[0])
    width_bottom = np.linalg.norm(rect[2] - rect[3])
    height_right = np.linalg.norm(rect[2] - rect[1])
    height_left = np.linalg.norm(rect[3] - rect[0])

    out_w = int(max(width_top, width_bottom))
    out_h = int(max(height_right, height_left))
    if out_w < 24 or out_h < 24:
        return None

    # Normalize output size for better downstream card reading quality.
    # MTG card aspect ratio is close to 2.5:3.5 ~= 0.714 (width/height).
    target_h = max(128, int(output_height))
    target_w = max(96, int(round(target_h * (2.5 / 3.5))))

    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (target_w, target_h), flags=cv2.INTER_LANCZOS4)


def enhance_crop(crop: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return crop
    blurred = cv2.GaussianBlur(crop, (0, 0), 1.0)
    return cv2.addWeighted(crop, 1.35, blurred, -0.35, 0)


def ahash(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).astype(np.uint8).reshape(-1)
    return bits


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def polygon_to_bbox(poly: np.ndarray) -> tuple[float, float, float, float]:
    pts = poly.reshape(4, 2)
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return x1, y1, x2, y2


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def crop_sharpness_score(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_duplicate_saved(
    sig: np.ndarray,
    bbox: tuple[float, float, float, float],
    saved_cards: list[SavedCard],
    hash_threshold: int,
    dedupe_iou: float,
) -> bool:
    for card in saved_cards:
        prev_bbox = (card.cx - card.w / 2, card.cy - card.h / 2, card.cx + card.w / 2, card.cy + card.h / 2)
        spatial_iou = bbox_iou(bbox, prev_bbox)
        visual_dist = hamming_distance(sig, card.hash_bits)
        if spatial_iou >= dedupe_iou or visual_dist <= hash_threshold:
            return True
    return False


def update_crop_candidates(
    frame: np.ndarray,
    polys: np.ndarray,
    confs: np.ndarray,
    frame_idx: int,
    min_conf: float,
    output_height: int,
    margin: float,
    enhance_mode: str,
    hash_threshold: int,
    track_iou: float,
    saved_cards: list[SavedCard],
    dedupe_iou: float,
    candidates: list[CropCandidate],
) -> None:
    for poly, conf in zip(polys, confs):
        conf_f = float(conf)
        if conf_f < min_conf:
            continue

        crop = crop_from_polygon(frame, poly, output_height=output_height, margin=margin)
        if crop is None:
            continue
        crop = enhance_crop(crop, enhance_mode)

        sig = ahash(crop)
        x1, y1, x2, y2 = polygon_to_bbox(poly)
        bbox = (x1, y1, x2, y2)
        if is_duplicate_saved(sig, bbox, saved_cards, hash_threshold=hash_threshold, dedupe_iou=dedupe_iou):
            continue

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        sharpness = crop_sharpness_score(crop)
        score = sharpness * (0.5 + 0.5 * conf_f)

        best_idx = -1
        best_match_score = -1.0
        for idx, cand in enumerate(candidates):
            cand_bbox = (cand.cx - cand.w / 2, cand.cy - cand.h / 2, cand.cx + cand.w / 2, cand.cy + cand.h / 2)
            iou = bbox_iou(bbox, cand_bbox)
            ham = hamming_distance(sig, cand.hash_bits)
            if iou >= track_iou or ham <= hash_threshold:
                match_score = iou - (ham / 64.0)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_idx = idx

        if best_idx == -1:
            candidates.append(
                CropCandidate(
                    hash_bits=sig,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    best_crop=crop,
                    best_conf=conf_f,
                    best_sharpness=sharpness,
                    best_score=score,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    seen_count=1,
                )
            )
            continue

        cand = candidates[best_idx]
        cand.last_frame = frame_idx
        cand.seen_count += 1
        cand.cx = cx
        cand.cy = cy
        cand.w = w
        cand.h = h
        cand.hash_bits = sig
        if score > cand.best_score:
            cand.best_crop = crop
            cand.best_conf = conf_f
            cand.best_sharpness = sharpness
            cand.best_score = score


def flush_ready_crop_candidates(
    output_dir: Path,
    jpeg_quality: int,
    frame_idx: int,
    settle_frames: int,
    min_observations: int,
    min_sharpness: float,
    hash_threshold: int,
    dedupe_iou: float,
    saved_cards: list[SavedCard],
    candidates: list[CropCandidate],
    flush_all: bool = False,
) -> int:
    saved = 0
    remaining: list[CropCandidate] = []
    for cand in candidates:
        idle_frames = frame_idx - cand.last_frame
        age_frames = frame_idx - cand.first_frame
        # Save when candidate has matured, even if still visible/moving.
        ready = flush_all or (
            cand.seen_count >= min_observations
            and (idle_frames >= settle_frames or age_frames >= settle_frames)
        )
        if not ready:
            remaining.append(cand)
            continue

        if cand.best_sharpness < min_sharpness:
            # Keep the candidate alive so it can improve with future, sharper frames.
            remaining.append(cand)
            continue

        bbox = (cand.cx - cand.w / 2, cand.cy - cand.h / 2, cand.cx + cand.w / 2, cand.cy + cand.h / 2)
        if is_duplicate_saved(cand.hash_bits, bbox, saved_cards, hash_threshold=hash_threshold, dedupe_iou=dedupe_iou):
            # Already represented in saved set; drop this candidate.
            continue

        saved_cards.append(SavedCard(hash_bits=cand.hash_bits, cx=cand.cx, cy=cand.cy, w=cand.w, h=cand.h))
        filename = (
            f"frame{cand.last_frame:06d}_seen{cand.seen_count:02d}_"
            f"conf{cand.best_conf:.2f}_sharp{cand.best_sharpness:.1f}.jpg"
        )
        cv2.imwrite(
            str(output_dir / filename),
            cand.best_crop,
            [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, jpeg_quality)))],
        )
        saved += 1

    candidates[:] = remaining
    return saved


def main() -> None:
    args = parse_args()
    if args.infer_every < 1:
        raise ValueError("--infer-every must be >= 1")
    if args.record_every < 1:
        raise ValueError("--record-every must be >= 1")
    if args.crop_hash_threshold < 0:
        raise ValueError("--crop-hash-threshold must be >= 0")
    if args.crop_settle_frames < 0:
        raise ValueError("--crop-settle-frames must be >= 0")
    if args.crop_debug_every < 0:
        raise ValueError("--crop-debug-every must be >= 0")
    if args.crop_min_observations < 1:
        raise ValueError("--crop-min-observations must be >= 1")
    if not (0.0 <= args.crop_margin <= 0.2):
        raise ValueError("--crop-margin must be between 0.0 and 0.2")
    if not (0.0 <= args.crop_track_iou <= 1.0):
        raise ValueError("--crop-track-iou must be between 0.0 and 1.0")
    if not (0.0 <= args.crop_dedupe_iou <= 1.0):
        raise ValueError("--crop-dedupe-iou must be between 0.0 and 1.0")
    if args.crop_output_height < 128:
        raise ValueError("--crop-output-height must be >= 128")
    if not (1 <= args.crop_jpeg_quality <= 100):
        raise ValueError("--crop-jpeg-quality must be in [1, 100]")
    if not (0.25 <= args.inference_scale <= 1.0):
        raise ValueError("--inference-scale must be between 0.25 and 1.0")

    model_path = resolve_model_path(args.model)

    model = YOLO(str(model_path))
    cap = open_camera(args.camera_index, args.width, args.height)
    writer, output_path = make_video_writer(args)
    crops_dir = Path(args.crops_dir).expanduser().resolve()
    if args.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    fps_samples: deque[float] = deque(maxlen=30)
    frame_idx = 0
    last_polys = np.empty((0, 4, 2), dtype=np.float32)
    last_confs = np.empty((0,), dtype=np.float32)
    saved_cards: list[SavedCard] = []
    crop_candidates: list[CropCandidate] = []
    print("[INFO] Live webcam detection started. Press 'q' to quit.")
    print(
        "[INFO] Runtime profile: "
        f"imgsz={args.imgsz} infer_every={args.infer_every} "
        f"inference_scale={args.inference_scale:.2f} max_det={args.max_det}"
    )
    if output_path is not None:
        print(f"[INFO] Recording annotated stream to: {output_path} (record_every={args.record_every})")
    if args.save_crops:
        print(
            f"[INFO] Saving unique card crops to: {crops_dir} "
            f"(min_conf={args.crop_min_conf:.2f}, h={args.crop_output_height}, q={args.crop_jpeg_quality}, "
            f"margin={args.crop_margin:.3f}, dedupe_iou={args.crop_dedupe_iou:.2f}, hash_thr={args.crop_hash_threshold}, "
            f"track_iou={args.crop_track_iou:.2f}, settle={args.crop_settle_frames}, min_obs={args.crop_min_observations}, "
            f"min_sharp={args.crop_min_sharpness:.1f})"
        )

    # Read one frame to report actual negotiated camera dimensions.
    ok_probe, probe = cap.read()
    if ok_probe:
        probe = apply_rotation(probe, args.rotate)
        actual_h, actual_w = probe.shape[:2]
        req = f"{args.width}x{args.height}"
        got = f"{actual_w}x{actual_h}"
        print(f"[INFO] Camera resolution requested={req} actual={got} rotate={args.rotate}")
    else:
        print("[WARN] Unable to read probe frame for resolution diagnostics")

    try:
        while True:
            start = time.perf_counter()
            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame from webcam")
                continue
            frame = apply_rotation(frame, args.rotate)

            should_infer = (frame_idx % args.infer_every) == 0
            if should_infer:
                if args.inference_scale < 1.0:
                    infer_w = max(64, int(frame.shape[1] * args.inference_scale))
                    infer_h = max(64, int(frame.shape[0] * args.inference_scale))
                    infer_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                    scale_x = frame.shape[1] / float(infer_w)
                    scale_y = frame.shape[0] / float(infer_h)
                else:
                    infer_frame = frame
                    scale_x = 1.0
                    scale_y = 1.0

                result = model.predict(
                    source=infer_frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    max_det=args.max_det,
                    verbose=False,
                )[0]
                last_polys, last_confs = extract_polys_confs(result, scale_x=scale_x, scale_y=scale_y)
                if args.save_crops:
                    update_crop_candidates(
                        frame=frame,
                        polys=last_polys,
                        confs=last_confs,
                        frame_idx=frame_idx,
                        min_conf=args.crop_min_conf,
                        output_height=args.crop_output_height,
                        margin=args.crop_margin,
                        enhance_mode=args.crop_enhance,
                        hash_threshold=args.crop_hash_threshold,
                        track_iou=args.crop_track_iou,
                        saved_cards=saved_cards,
                        dedupe_iou=args.crop_dedupe_iou,
                        candidates=crop_candidates,
                    )
                    new_saves = flush_ready_crop_candidates(
                        output_dir=crops_dir,
                        jpeg_quality=args.crop_jpeg_quality,
                        frame_idx=frame_idx,
                        settle_frames=args.crop_settle_frames,
                        min_observations=args.crop_min_observations,
                        min_sharpness=args.crop_min_sharpness,
                        hash_threshold=args.crop_hash_threshold,
                        dedupe_iou=args.crop_dedupe_iou,
                        saved_cards=saved_cards,
                        candidates=crop_candidates,
                    )
                    if new_saves:
                        print(f"[INFO] Saved {new_saves} new crop(s). Total unique={len(saved_cards)}")
                    elif args.crop_debug_every and (frame_idx % args.crop_debug_every) == 0:
                        best_sharp = max((c.best_sharpness for c in crop_candidates), default=0.0)
                        print(
                            f"[DEBUG] crop_candidates={len(crop_candidates)} saved={len(saved_cards)} "
                            f"best_sharp={best_sharp:.1f} min_sharp={args.crop_min_sharpness:.1f}"
                        )

            detections = draw_cached_detections(frame, last_polys, last_confs, args.line_thickness)

            elapsed = time.perf_counter() - start
            if elapsed > 0:
                fps_samples.append(1.0 / elapsed)
            avg_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0

            overlay = (
                f"FPS: {avg_fps:.1f} | Detections: {detections} | "
                f"conf>={args.conf:.2f} | every={args.infer_every} | scale={args.inference_scale:.2f}"
            )
            cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2, cv2.LINE_AA)

            if writer is not None and (frame_idx % args.record_every) == 0:
                writer.write(frame)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        if args.save_crops:
            flushed = flush_ready_crop_candidates(
                output_dir=crops_dir,
                jpeg_quality=args.crop_jpeg_quality,
                frame_idx=frame_idx,
                settle_frames=args.crop_settle_frames,
                min_observations=1,
                min_sharpness=args.crop_min_sharpness,
                hash_threshold=args.crop_hash_threshold,
                dedupe_iou=args.crop_dedupe_iou,
                saved_cards=saved_cards,
                candidates=crop_candidates,
                flush_all=True,
            )
            if flushed:
                print(f"[INFO] Final flush saved {flushed} crop(s). Total unique={len(saved_cards)}")
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
