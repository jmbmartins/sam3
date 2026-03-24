import torch
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image

import supervision as sv

from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video

import imageio.v3 as iio  # pip install "imageio[ffmpeg]"


# --------- CONFIGURE THESE ---------
INPUT_FOLDER  = "/home/evox5090ia/datasets/suma_saojoaodamadeira/videos"   # folder with .mp4 files
TEXT_PROMPT   = "road pothole"
OUTPUT_FOLDER = "/home/evox5090ia/datasets/suma_saojoaodamadeira/videos/sam3_outputs"  # only videos WITH detections land here

CONF_THRESH = 0.40
OUTPUT_FPS  = 10
# -----------------------------------


COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])


def annotate(
    image: Image.Image,
    detections: sv.Detections,
    label: Optional[str] = None,
) -> Image.Image:
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)

    mask_annotator  = sv.MaskAnnotator(color=COLOR, color_lookup=sv.ColorLookup.INDEX, opacity=0.6)
    box_annotator   = sv.BoxAnnotator(color=COLOR, color_lookup=sv.ColorLookup.INDEX, thickness=1)
    label_annotator = sv.LabelAnnotator(
        color=COLOR, color_lookup=sv.ColorLookup.INDEX,
        text_scale=text_scale, text_padding=5,
        text_color=sv.Color.BLACK, text_thickness=1,
    )

    annotated = image.copy()
    annotated = mask_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)

    if label:
        labels = [
            f"id={tid} | {label} {conf:.2f}"
            for tid, conf in zip(detections.class_id, detections.confidence)
        ]
        annotated = label_annotator.annotate(annotated, detections, labels)

    return annotated


def frame_outputs_to_detections(
    frame_outputs: Dict[str, Any],
    conf_thresh: float = 0.05,
) -> Optional[sv.Detections]:
    if frame_outputs is None:
        return None

    boxes      = frame_outputs["boxes"]
    scores     = frame_outputs["scores"]
    masks      = frame_outputs["masks"]
    object_ids = frame_outputs["object_ids"]

    keep = scores > conf_thresh
    if keep.sum() == 0:
        return None

    boxes_np   = boxes[keep].detach().cpu().numpy().astype(np.float32)
    scores_np  = scores[keep].detach().cpu().numpy().astype(np.float32)
    masks_np   = masks[keep].detach().cpu().numpy()
    obj_ids_np = object_ids[keep].detach().cpu().numpy().astype(np.int32)

    masks_bin = masks_np > 0.5 if masks_np.dtype != bool else masks_np

    return sv.Detections(
        xyxy=boxes_np,
        mask=masks_bin,
        confidence=scores_np,
        class_id=obj_ids_np,
    )


def process_video(
    video_path: Path,
    model,
    processor,
    device: torch.device,
    dtype: torch.dtype,
    output_folder: Path,
) -> bool:
    """
    Run SAM3 on a single video.
    Returns True if at least one detection was found and the output was saved.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")

    video_frames, _ = load_video(str(video_path))
    num_frames = len(video_frames)
    print(f"  Frames: {num_frames}")

    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=TEXT_PROMPT,
    )

    outputs_per_frame: Dict[int, Dict[str, Any]] = {}
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=num_frames,
    ):
        processed = processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed

    # ── Check whether any frame produced detections ──────────────────────────
    any_detection = False
    for frame_out in outputs_per_frame.values():
        det = frame_outputs_to_detections(frame_out, CONF_THRESH)
        if det is not None and len(det) > 0:
            any_detection = True
            break

    if not any_detection:
        print(f"  No detections found — skipping output.")
        return False

    # ── Build annotated frames ────────────────────────────────────────────────
    frames_to_write: List[np.ndarray] = []
    for idx, frame in enumerate(video_frames):
        frame_pil = frame.convert("RGB") if isinstance(frame, Image.Image) \
                    else Image.fromarray(frame).convert("RGB")

        frame_out = outputs_per_frame.get(idx)
        if not frame_out:
            frames_to_write.append(np.array(frame_pil))
            continue

        det = frame_outputs_to_detections(frame_out, CONF_THRESH)
        if det is None or len(det) == 0:
            frames_to_write.append(np.array(frame_pil))
            continue

        annotated = annotate(frame_pil, det, label=TEXT_PROMPT)
        frames_to_write.append(np.array(annotated))

    # ── Write output video ────────────────────────────────────────────────────
    out_path = output_folder / video_path.name
    iio.imwrite(out_path, frames_to_write, fps=OUTPUT_FPS, codec="libx264")
    print(f"  Saved annotated video → {out_path}")
    return True


def main():
    input_folder  = Path(INPUT_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    mp4_files = sorted(input_folder.glob("*.mp4"))
    if not mp4_files:
        raise RuntimeError(f"No .mp4 files found in: {input_folder}")

    print(f"Found {len(mp4_files)} MP4 file(s) in {input_folder}")
    print(f"Text prompt : '{TEXT_PROMPT}'")
    print(f"Output folder (detections only): {output_folder}\n")

    output_folder.mkdir(parents=True, exist_ok=True)

    # ── Load model once, reuse for all videos ─────────────────────────────────
    accelerator = Accelerator()
    device = accelerator.device
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    print("Loading SAM3 Video model...")
    model     = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    print("Model loaded.\n")

    # ── Process every video ───────────────────────────────────────────────────
    saved, skipped = 0, 0
    for video_path in mp4_files:
        try:
            was_saved = process_video(
                video_path, model, processor, device, dtype, output_folder
            )
            if was_saved:
                saved += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"  ERROR processing {video_path.name}: {exc}")
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Finished.  Saved: {saved}  |  Skipped (no detections / error): {skipped}")


if __name__ == "__main__":
    main()