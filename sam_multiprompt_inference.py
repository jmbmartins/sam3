import torch
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image

import supervision as sv

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --------- CONFIGURE THESE ---------
IMAGE_PATH = "/home/evox5090ia/sam3/assets/images/image1.png"
OUTPUT_PATH = "/home/evox5090ia/sam3/sam3_multi_outputs/image1_output.png"

# multiple prompts to try for "overflow" situations
TEXT_PROMPTS: List[str] = [
    "trash on the ground near the container",
    "cardboard on the ground",
    "cardboard boxes on the ground",
    "flattened cardboard near the container",
    "garbage bag on the ground",
    "white plastic bag on the ground",
]

LABEL_NAME = "overflow trash"   # what will be shown on the boxes
CONF_THRESH = 0.05
# -----------------------------------

# Color palette for Supervision
COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])


def annotate(image: Image.Image, detections: sv.Detections, label: Optional[str] = None) -> Image.Image:
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)

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


def sam_output_to_detections(masks, boxes, scores) -> sv.Detections:
    """
    Convert SAM3 output into sv.Detections (xyxy, mask, confidence, class_id).
    """
    # Convert to numpy
    boxes_np = boxes.detach().cpu().numpy().astype(np.float32)   # (N, 4)
    scores_np = scores.detach().cpu().numpy().astype(np.float32) # (N,)

    masks_np = masks.detach().cpu().numpy()  # could be (N, H, W) or (N, 1, H, W) etc.

    # Normalize mask shape to (N, H, W)
    if masks_np.ndim == 4:
        # Assume (N, 1, H, W) or (N, H, W, 1)
        if masks_np.shape[1] == 1:        # (N, 1, H, W)
            masks_np = masks_np[:, 0]
        elif masks_np.shape[-1] == 1:     # (N, H, W, 1)
            masks_np = masks_np[..., 0]
        else:
            raise RuntimeError(f"Unexpected 4D mask shape from SAM3: {masks_np.shape}")
    elif masks_np.ndim == 3:
        # Already (N, H, W)
        pass
    else:
        raise RuntimeError(f"Unexpected mask shape from SAM3: {masks_np.shape}")

    # Binarize masks (bool or 0/1)
    masks_bin = masks_np > 0.5  # (N, H, W) boolean

    # Dummy class_ids (all same class, since you're using text prompts)
    class_ids = np.zeros(len(scores_np), dtype=int)

    detections = sv.Detections(
        xyxy=boxes_np,
        mask=masks_bin,
        confidence=scores_np,
        class_id=class_ids,
    )

    return detections


def main():
    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Ensure output directory exists
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Torch device:", device)

    # Load SAM3 model
    print("Loading SAM3 model (this may download weights the first time)...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("Model loaded.\n")

    # Load image
    image = Image.open(img_path).convert("RGB")

    all_detections = []

    # ---- MULTI-PROMPT INFERENCE ----
    for prompt in TEXT_PROMPTS:
        print(f"\n=== Running text prompt: '{prompt}' ===")

        # For safety, reset image state for each prompt
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        print(f"Found {len(masks)} instances for prompt '{prompt}'")
        for i, (box, score) in enumerate(zip(boxes, scores)):
            print(f"  #{i}: score={float(score):.3f}, box={box.tolist()}")

        if len(masks) == 0:
            continue

        det = sam_output_to_detections(masks, boxes, scores)

        # filter by confidence
        det = det[det.confidence > CONF_THRESH]
        print(f"After confidence filtering (>{CONF_THRESH}): {len(det)} detections")

        if len(det) > 0:
            all_detections.append(det)

    if not all_detections:
        print("No detections for any prompt; nothing to annotate.")
        return

    # Merge detections from all prompts
    detections = sv.Detections.merge(all_detections)
    print(f"\nTotal merged detections: {len(detections)}")

    # Annotate image
    annotated = annotate(image, detections, label=LABEL_NAME)

    # Show and save
    annotated.show()  # pops up an image viewer window
    annotated.save(out_path)
    print(f"Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()
