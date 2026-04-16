import shutil
import subprocess
from pathlib import Path
from typing import Dict, Set

import torch
import numpy as np
from PIL import Image
import supervision as sv
import cv2

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ================= CONFIG =================
ROOT_DIR = Path("/home/evox5090ia/Downloads/videos_demonstracao")

PROMPTS: Dict[int, str] = {
    0: "person",
    1: "license plate",
}

BLACKOUT_CLASS_IDS: Set[int] = {0, 1}
CONF_THRESH = 0.05
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}

# How often to run SAM3 (every N frames). Detections from the last
# inference are reused in between to save compute.
# Set to 1 to run on every single frame (slowest but most accurate).
INFERENCE_EVERY_N_FRAMES = 1
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


def blackout_by_class(
    img_np: np.ndarray,
    detections: sv.Detections,
    class_ids_to_blackout: Set[int],
) -> np.ndarray:
    """Apply black mask in-place on a BGR numpy frame. Returns modified array."""
    if detections is None or detections.mask is None or len(detections) == 0:
        return img_np

    sel = np.isin(detections.class_id, list(class_ids_to_blackout))
    det_sel = detections[sel]

    if len(det_sel) == 0 or det_sel.mask is None:
        return img_np

    masks = det_sel.mask
    union = masks if masks.ndim == 2 else np.any(masks, axis=0)
    img_np[union] = 0
    return img_np


def run_sam3_on_frame(
    processor: Sam3Processor,
    frame_rgb: np.ndarray,
) -> sv.Detections | None:
    """Run SAM3 text-prompted inference on a single RGB numpy frame."""
    image = Image.fromarray(frame_rgb)
    all_dets = []

    for class_id, prompt in PROMPTS.items():
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)

        if not output or len(output.get("masks", [])) == 0:
            continue

        det = sam_output_to_detections(
            output["masks"], output["boxes"], output["scores"], class_id
        )
        det = det[det.confidence > CONF_THRESH]

        if len(det) > 0:
            all_dets.append(det)

    if not all_dets:
        return None

    return sv.Detections.merge(all_dets)


def has_audio_stream(video_path: Path) -> bool:
    """Check if the video file has an audio stream using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() != ""


def mux_audio(original_video: Path, silent_video: Path, output_video: Path) -> bool:
    """Copy audio from original into the anonymised video using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(silent_video),
            "-i", str(original_video),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_video),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [WARN] ffmpeg audio mux failed: {result.stderr.strip()}")
        return False
    return True


def process_video(
    video_path: Path,
    out_dir: Path,
    processor: Sam3Processor,
    tmp_dir: Path,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path.name}, skipping.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write to a temporary file first; audio mux will produce the final file.
    tmp_out = tmp_dir / (video_path.stem + "_tmp.mp4")
    final_out = out_dir / (video_path.stem + "_ano.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_out), fourcc, fps, (width, height))

    print(f"  Processing '{video_path.name}'  ({total_frames} frames @ {fps:.1f} fps)")

    last_detections: sv.Detections | None = None
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Run SAM3 every N frames; reuse last result in between.
        if frame_idx % INFERENCE_EVERY_N_FRAMES == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            last_detections = run_sam3_on_frame(processor, frame_rgb)

        frame_bgr = blackout_by_class(frame_bgr, last_detections, BLACKOUT_CLASS_IDS)
        writer.write(frame_bgr)

        if (frame_idx + 1) % 100 == 0:
            pct = (frame_idx + 1) / max(total_frames, 1) * 100
            print(f"    frame {frame_idx + 1}/{total_frames}  ({pct:.1f}%)")

        frame_idx += 1

    cap.release()
    writer.release()

    # ---- Audio mux ------------------------------------------------
    if has_audio_stream(video_path):
        print(f"  Muxing audio from original...")
        success = mux_audio(video_path, tmp_out, final_out)
        tmp_out.unlink(missing_ok=True)
        if not success:
            tmp_out.rename(final_out)
    else:
        tmp_out.rename(final_out)

    print(f"  Saved -> {final_out.name}\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    videos_dir = ROOT_DIR  # .MOV files sit directly in ROOT_DIR
    out_dir    = ROOT_DIR / "videos_ano"
    tmp_dir    = ROOT_DIR / "_tmp"

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_paths:
        print(f"No video files found in {videos_dir}")
        return

    print(f"Found {len(video_paths)} video(s).\n")

    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM3 loaded.\n")

    for i, vp in enumerate(video_paths):
        print(f"[{i + 1}/{len(video_paths)}] {vp.name}")
        process_video(vp, out_dir, processor, tmp_dir)

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("All videos processed.")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()