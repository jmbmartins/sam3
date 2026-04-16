from pathlib import Path
from typing import Dict, Set

import torch
import numpy as np
from PIL import Image
import cv2
import supervision as sv

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ================= CONFIG =================
INPUT_VIDEO  = Path("/home/evox5090ia/suma_demo/1_direita.mp4")
OUTPUT_VIDEO = Path("/home/evox5090ia/suma_demo/1_direita_ano.mp4")

PROMPTS: Dict[int, str] = {
    0: "person",
    1: "license plate",
}

BLACKOUT_CLASS_IDS: Set[int] = {0, 1}
CONF_THRESH = 0.05
# =========================================


def sam_output_to_detections(masks, boxes, scores, class_id: int) -> sv.Detections:
    boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
    scores_np = scores.detach().cpu().numpy().astype(np.float32)
    masks_np = masks.detach().cpu().numpy()

    if masks_np.ndim == 4:
        if masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        elif masks_np.shape[-1] == 1:
            masks_np = masks_np[..., 0]
        else:
            raise RuntimeError(f"Unexpected mask shape {masks_np.shape}")

    masks_bin = masks_np > 0.5
    class_ids = np.full(len(scores_np), class_id, dtype=int)

    return sv.Detections(
        xyxy=boxes_np,
        mask=masks_bin,
        confidence=scores_np,
        class_id=class_ids,
    )


def blackout_by_class(image: Image.Image, detections: sv.Detections, class_ids_to_blackout: set[int]) -> Image.Image:
    if detections.mask is None or len(detections) == 0:
        return image

    img_np = np.array(image)
    sel = np.isin(detections.class_id, list(class_ids_to_blackout))
    det_sel = detections[sel]

    if len(det_sel) == 0 or det_sel.mask is None:
        return image

    masks = det_sel.mask
    union = masks if masks.ndim == 2 else np.any(masks, axis=0)

    img_np[union] = 0
    return Image.fromarray(img_np)


def main():
    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_VIDEO}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

    print("Loading SAM3 model...")
    model     = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM3 loaded.\n")

    print(f"Processing: {INPUT_VIDEO.name}")
    print(f"  Resolution : {width}x{height}  |  FPS: {fps:.2f}  |  Frames: {total_frames}")

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        all_dets = []
        for class_id, prompt in PROMPTS.items():
            state  = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=prompt)

            if not output or len(output.get("masks", [])) == 0:
                continue

            det = sam_output_to_detections(output["masks"], output["boxes"], output["scores"], class_id)
            det = det[det.confidence > CONF_THRESH]

            if len(det) > 0:
                all_dets.append(det)

        if all_dets:
            detections = sv.Detections.merge(all_dets)
            image      = blackout_by_class(image, detections, BLACKOUT_CLASS_IDS)

        out_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        writer.write(out_bgr)

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print(f"  [{frame_idx}/{total_frames}] frames processed...")

    cap.release()
    writer.release()
    print(f"\nDone. Output saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()