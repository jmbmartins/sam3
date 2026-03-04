import torch
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
from PIL import Image

import supervision as sv

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================
#         CONFIG
# =========================
INPUT_DIR = Path("/home/evox5090ia/datasets/veolia_2025-10-27_10-11-08_yolo/images")  # folder with images
OUTPUT_DIR = Path("/datasets/suma_saojoaodamadeira/waste_collection_worker_veolia")

# Example prompt for graffiti:
# TEXT_PROMPT = "spray paint graffiti or marker scribbles on the waste container surface; exclude logos, labels,
# stickers"

TEXT_PROMPT = "waste collection worker"


# If you want fissures instead, try:
# TEXT_PROMPT = "hairline crack or fissure on the plastic container wall; exclude seams, joints, shadows, scratches"

CONF_THRESH = 0.05
RECURSIVE = False  # set True if you want to scan subfolders too

# Save only when at least 1 detection passes threshold
SAVE_ONLY_WHEN_DETECTED = True

# Mask threshold for binarization (SAM3 mask logits/probabilities)
MASK_BIN_THRESH = 0.5
# =========================


# ----------------- ANNOTATION CONFIG -----------------
COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(folder: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p
    else:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def annotate(image: Image.Image, detections: sv.Detections, label: Optional[str] = None) -> Image.Image:
    mask_annotator = sv.MaskAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.6,
    )
    box_annotator = sv.BoxAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.INDEX,
        thickness=1,
    )
    label_annotator = sv.LabelAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=0.4,
        text_padding=5,
        text_color=sv.Color.BLACK,
        text_thickness=1,
    )

    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(annotated_image, detections)
    annotated_image = box_annotator.annotate(annotated_image, detections)

    if label:
        labels = [f"{label} {confidence:.2f}" for confidence in detections.confidence]
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)

    return annotated_image


def sam_output_to_detections(masks, boxes, scores, mask_bin_thresh: float = 0.5) -> sv.Detections:
    """
    Convert SAM3 output into sv.Detections (xyxy, mask, confidence, class_id).
    """
    boxes_np = boxes.detach().cpu().numpy().astype(np.float32)     # (N, 4)
    scores_np = scores.detach().cpu().numpy().astype(np.float32)   # (N,)
    masks_np = masks.detach().cpu().numpy()

    # Normalize mask shape to (N, H, W)
    if masks_np.ndim == 4:
        if masks_np.shape[1] == 1:            # (N, 1, H, W)
            masks_np = masks_np[:, 0]
        elif masks_np.shape[-1] == 1:         # (N, H, W, 1)
            masks_np = masks_np[..., 0]
        else:
            raise RuntimeError(f"Unexpected 4D mask shape from SAM3: {masks_np.shape}")
    elif masks_np.ndim != 3:
        raise RuntimeError(f"Unexpected mask shape from SAM3: {masks_np.shape}")

    masks_bin = masks_np > mask_bin_thresh
    class_ids = np.zeros(len(scores_np), dtype=int)

    return sv.Detections(
        xyxy=boxes_np,
        mask=masks_bin,
        confidence=scores_np,
        class_id=class_ids,
    )


def safe_output_name(input_path: Path) -> str:
    # Force PNG outputs for consistent overlay saving
    return f"{input_path.stem}_sam3.png"


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Torch device: {device}")
    print("Loading SAM3 model (may download weights first time)...")

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    print("Model loaded.")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Prompt: {TEXT_PROMPT}")
    print(f"Confidence threshold: {CONF_THRESH}")
    print(f"Recursive: {RECURSIVE}")
    print(f"Save only when detected: {SAVE_ONLY_WHEN_DETECTED}\n")

    total = 0
    detected_imgs = 0
    saved = 0

    for img_path in iter_images(INPUT_DIR, recursive=RECURSIVE):
        total += 1

        try:
            image = Image.open(img_path).convert("RGB")

            # Inference
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=TEXT_PROMPT)

            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]

            if len(masks) == 0:
                print(f"[NO DET] {img_path.name} (0 instances)")
                if not SAVE_ONLY_WHEN_DETECTED:
                    out_file = OUTPUT_DIR / safe_output_name(img_path)
                    image.save(out_file)
                    saved += 1
                continue

            detections = sam_output_to_detections(masks, boxes, scores, mask_bin_thresh=MASK_BIN_THRESH)
            detections = detections[detections.confidence > CONF_THRESH]

            if len(detections) == 0:
                max_score = float(scores.max().detach().cpu()) if hasattr(scores, "max") else float(np.max(scores))
                print(f"[NO DET] {img_path.name} (instances={len(masks)} max_score={max_score:.3f} < thr)")
                if not SAVE_ONLY_WHEN_DETECTED:
                    out_file = OUTPUT_DIR / safe_output_name(img_path)
                    image.save(out_file)
                    saved += 1
                continue

            detected_imgs += 1

            # Annotate and save only when detected
            annotated = annotate(image, detections, label=TEXT_PROMPT)
            out_file = OUTPUT_DIR / safe_output_name(img_path)
            annotated.save(out_file)
            saved += 1

            best = float(detections.confidence.max())
            print(f"[DETECTED] {img_path.name} -> {out_file.name} (n={len(detections)} best={best:.3f})")


        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print("\n--- Summary ---")
    print(f"Images processed: {total}")
    print(f"Images with detections: {detected_imgs}")
    print(f"Outputs saved: {saved}")


if __name__ == "__main__":
    main()