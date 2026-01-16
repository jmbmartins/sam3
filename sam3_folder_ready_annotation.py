import torch
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

import supervision as sv

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ================= CONFIG =================
ROOT_DIR = Path("/home/evox5090ia/sumasaojoao/2025-12-22_08-45-47_for_ann/joaojoaq")  # <-- ajusta se preciso

PROMPTS: Dict[int, str] = {
    0: "person",
    1: "license plate"
}

CONF_THRESH = 0.05
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Output: pasta ao lado com sufixo _blackout, mantendo estrutura relativa
OUTPUT_ROOT = ROOT_DIR.parent / f"{ROOT_DIR.name}_blackout"
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


def blackout_detections(image: Image.Image, detections: sv.Detections) -> Image.Image:
    if detections.mask is None or len(detections) == 0:
        return image

    img_np = np.array(image)
    masks = detections.mask
    union = masks if masks.ndim == 2 else np.any(masks, axis=0)
    img_np[union] = 0
    return Image.fromarray(img_np)


def find_frames_dirs(root: Path) -> List[Path]:
    """
    Finds every directory named 'frames' under root.
    Each 'frames' directory is treated as one video sequence.
    """
    return sorted(p for p in root.rglob("frames") if p.is_dir())


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM3 loaded.\n")

    frames_dirs = find_frames_dirs(ROOT_DIR)
    if not frames_dirs:
        print("No 'frames' directories found under:", ROOT_DIR)
        return

    print(f"Found {len(frames_dirs)} frames directories.\n")

    for frames_dir in frames_dirs:
        # Example: ROOT/direita/88/frames
        # We want output to be: OUTPUT_ROOT/direita/88/frames (same relative path)
        rel = frames_dir.relative_to(ROOT_DIR)           # direita/88/frames
        out_frames_dir = OUTPUT_ROOT / rel               # ..._blackout/direita/88/frames
        out_frames_dir.mkdir(parents=True, exist_ok=True)

        # For logging: use parent folder name as "video id" (e.g., 88) plus side (direita)
        video_tag = str(rel.parent)  # direita/88

        frame_paths = sorted(
            p for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"[{video_tag}] {len(frame_paths)} frames")

        for frame_path in frame_paths:
            out_path = out_frames_dir / frame_path.name

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

            # If no detections, keep original frame (so video length stays consistent)
            if not all_detections:
                image.save(out_path)
                continue

            detections = sv.Detections.merge(all_detections)
            blacked = blackout_detections(image, detections)
            blacked.save(out_path)

        print(f"[{video_tag}] done\n")

    print("All sequences processed.")
    print("Output at:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
