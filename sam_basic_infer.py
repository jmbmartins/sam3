import torch
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import supervision as sv

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --------- CONFIGURE THESE ---------
IMAGE_PATH = "/home/evox5090ia/sam3/assets/images/image2.png"
TEXT_PROMPT = "person"
OUTPUT_PATH = "/home/evox5090ia/sam3/sam3_annotated.png"
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
        opacity=0.6
    )
    box_annotator = sv.BoxAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.INDEX,
        thickness=1
    )
    label_annotator = sv.LabelAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=0.4,
        text_padding=5,
        text_color=sv.Color.BLACK,
        text_thickness=1
    )

    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(annotated_image, detections)
    annotated_image = box_annotator.annotate(annotated_image, detections)

    if label:
        labels = [
            f"{label} {confidence:.2f}"
            for confidence in detections.confidence
        ]
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)

    return annotated_image


def sam_output_to_detections(masks, boxes, scores) -> sv.Detections:
    """
    Convert SAM3 output into sv.Detections (xyxy, mask, confidence, class_id).
    """
    # Convert to numpy
    boxes_np = boxes.detach().cpu().numpy().astype(np.float32)  # (N, 4)
    scores_np = scores.detach().cpu().numpy().astype(np.float32)  # (N,)

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

    # Dummy class_ids (all same class, since you're using a text prompt)
    class_ids = np.zeros(len(scores_np), dtype=int)

    detections = sv.Detections(
        xyxy=boxes_np,
        mask=masks_bin,
        confidence=scores_np,
        class_id=class_ids
    )

    return detections


def main():
    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Torch device:", device)

    # Load SAM3 model
    print("Loading SAM3 model (this may download weights the first time)...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("Model loaded.\n")

    # Load image
    image = Image.open(img_path).convert("RGB")

    # Inference with text prompt
    inference_state = processor.set_image(image)
    print(f"Running text prompt: '{TEXT_PROMPT}'")
    output = processor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    print(f"\nFound {len(masks)} instances for prompt '{TEXT_PROMPT}'")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  #{i}: score={float(score):.3f}, box={box.tolist()}")

    if len(masks) == 0:
        print("No detections, nothing to annotate.")
        return

    # Convert SAM3 output to Supervision detections
    detections = sam_output_to_detections(masks, boxes, scores)

    # (Optional) filter by confidence, e.g. > 0.5
    detections = detections[detections.confidence > 0.5]
    print(f"\nAfter confidence filtering (>0.5): {len(detections)} detections")

    # Annotate image
    annotated = annotate(image, detections, label=TEXT_PROMPT)

    # Show and save
    annotated.show()  # pops up an image viewer window
    annotated.save(OUTPUT_PATH)
    print(f"Annotated image saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
