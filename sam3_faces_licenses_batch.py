import shutil
from pathlib import Path
from typing import Dict, Set

import torch
import numpy as np
from PIL import Image
import supervision as sv

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ================= CONFIG =================
ROOT_DIR = Path("/home/evox5090ia/rotasinprogress_sumasaojoao/2026-03-18_08-26-45/video/direita")

PROMPTS: Dict[int, str] = {
    0: "person",
    1: "license plate",
}

BLACKOUT_CLASS_IDS: Set[int] = {0, 1}
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
    if detections.mask is None or len(detections) == 0:
        return image

    img_np = np.array(image)
    sel = np.isin(detections.class_id, list(class_ids_to_blackout))
    det_sel = detections[sel]

    if len(det_sel) == 0 or det_sel.mask is None:
        return image

    masks = det_sel.mask  # (N, H, W) or (H, W)
    union = masks if masks.ndim == 2 else np.any(masks, axis=0)

    img_np[union] = 0
    return Image.fromarray(img_np)


def load_rgb(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM3 loaded.\n")

    # each "videoid" dir: .../direita/<id>/
    video_dirs = sorted(d for d in ROOT_DIR.iterdir() if d.is_dir())

    for video_dir in video_dirs:
        frames_dir = video_dir / "frames"
        if not frames_dir.is_dir():
            continue

        # output alongside frames/
        frames_ano_dir = video_dir / "frames_ano"
        frames_ano_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted(
            p for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"[VIDEO {video_dir.name}] {len(frame_paths)} frames -> {frames_ano_dir}")

        for frame_path in frame_paths:
            out_path = frames_ano_dir / frame_path.name

            image = load_rgb(frame_path)
            if image is None:
                # if can't open, just copy the raw file
                try:
                    shutil.copy2(frame_path, out_path)
                except Exception as e:
                    print(f"[WARN] Failed to copy {frame_path} -> {out_path}: {e}")
                continue

            all_dets = []

            for class_id, prompt in PROMPTS.items():
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=prompt)

                if not output or len(output.get("masks", [])) == 0:
                    continue

                det = sam_output_to_detections(output["masks"], output["boxes"], output["scores"], class_id)
                det = det[det.confidence > CONF_THRESH]

                if len(det) > 0:
                    all_dets.append(det)

            if not all_dets:
                # keep 1:1 structure even without detections
                image.save(out_path)
                continue

            detections = sv.Detections.merge(all_dets)
            blacked = blackout_by_class(image, detections, BLACKOUT_CLASS_IDS)
            blacked.save(out_path)

        print(f"[VIDEO {video_dir.name}] done")

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()
