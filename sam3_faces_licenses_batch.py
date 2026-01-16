import torch
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Dict

import supervision as sv

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ================= CONFIG =================
ROOT_DIR = Path("/home/evox5090ia/sumasaojoao/2025-12-22_08-45-47_for_ann/joaojoaq/direita")
OUTPUT_DIR = ROOT_DIR.parent / "detections_blackout"

PROMPTS: Dict[int, str] = {
    0: "person",
    1: "license plate"
}

LABELS = {
    0: "person",
    1: "plate"
}

CONF_THRESH = 0.05
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# =========================================


def sam_output_to_detections(masks, boxes, scores, class_id: int) -> sv.Detections:
    boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
    scores_np = scores.detach().cpu().numpy().astype(np.float32)
    masks_np = masks.detach().cpu().numpy()

    # Normalize masks to (N, H, W)
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
    """
    Paint black all pixels covered by any mask whose detection class_id is in class_ids_to_blackout.
    """
    if detections.mask is None or len(detections) == 0:
        return image

    img_np = np.array(image)  # (H, W, 3), uint8

    sel = np.isin(detections.class_id, list(class_ids_to_blackout))
    det_sel = detections[sel]
    if len(det_sel) == 0 or det_sel.mask is None:
        return image

    masks = det_sel.mask  # (N, H, W) or (H, W)
    union = masks if masks.ndim == 2 else np.any(masks, axis=0)

    img_np[union] = 0
    return Image.fromarray(img_np)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading SAM3 model...")
    model = build_sam3_image_model()  # assumes it handles device internally; if not, see note below
    processor = Sam3Processor(model)
    print("SAM3 loaded.\n")

    video_dirs = sorted(d for d in ROOT_DIR.iterdir() if d.is_dir())

    for video_dir in video_dirs:
        video_id = video_dir.name
        frames_dir = video_dir / "frames"

        if not frames_dir.exists():
            continue

        frame_paths = sorted(
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"[VIDEO {video_id}] {len(frame_paths)} frames")

        for frame_path in frame_paths:
            frame_name = frame_path.stem
            out_path = OUTPUT_DIR / f"{video_id}_{frame_name}.png"

            try:
                image = Image.open(frame_path).convert("RGB")
            except Exception as e:
                print(f"Failed to load {frame_path}: {e}")
                continue

            all_detections = []

            for class_id, prompt in PROMPTS.items():
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=prompt)

                if len(output.get("masks", [])) == 0:
                    continue

                det = sam_output_to_detections(
                    output["masks"],
                    output["boxes"],
                    output["scores"],
                    class_id
                )

                det = det[det.confidence > CONF_THRESH]

                if len(det) > 0:
                    all_detections.append(det)

            if not all_detections:
                continue

            detections = sv.Detections.merge(all_detections)

            # Black out ONLY persons + plates (class 0 and 1)
            blacked = blackout_by_class(image, detections, {0, 1})
            blacked.save(out_path)

        print(f"[VIDEO {video_id}] done")

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()
