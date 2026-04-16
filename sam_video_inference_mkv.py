import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile
import shutil

import numpy as np
from PIL import Image

import supervision as sv

from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video

import imageio.v3 as iio  # pip install "imageio[ffmpeg]"
import subprocess

# --------- CONFIGURE THESE ---------
VIDEO_PATH = "/home/evox5090ia/Downloads/2026-03-03_15-25-16_traseira_11.mkv"  # Now supports MKV
TEXT_PROMPT = "operator without helmet?"
OUTPUT_PATH = "/home/evox5090ia/Downloads/output.mp4"

CONF_THRESH = 0.5
OUTPUT_FPS = 10

# ===== MEMORY OPTIMIZATION SETTINGS =====
# Strategy 1: Process video in chunks
CHUNK_SIZE = 100  # Process N frames at a time (adjust based on VRAM)
OVERLAP_FRAMES = 10  # Overlap between chunks for consistency

# Strategy 2: Reduce resolution if needed
MAX_RESOLUTION = None  # Set to (width, height) to downscale, e.g., (1280, 720), or None for original

# Strategy 3: Skip frames for very long videos
FRAME_SKIP = 1  # Process every Nth frame (1 = all frames, 2 = every other frame, etc.)

# Strategy 4: Limit tracking objects
MAX_TRACKED_OBJECTS = 10  # Reduce if you get OOM

# Strategy 5: Use CPU offloading
USE_CPU_OFFLOAD = True  # Move inference state to CPU between chunks
# ========================================


# Color palette for Supervision
COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])


def convert_mkv_to_mp4(mkv_path: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Convert MKV to MP4 using ffmpeg.
    Returns path to the temporary MP4 file.
    """
    print(f"Converting MKV to MP4: {mkv_path}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())

    temp_mp4 = output_dir / f"{mkv_path.stem}_temp.mp4"

    # FFmpeg command for fast conversion (copy streams if possible)
    cmd = [
        "ffmpeg",
        "-i", str(mkv_path),
        "-c:v", "libx264",  # Re-encode video if needed
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",  # Re-encode audio
        "-b:a", "128k",
        "-y",  # Overwrite if exists
        str(temp_mp4)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Conversion complete: {temp_mp4}\n")
        return temp_mp4
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise


def resize_frames(frames: List[Image.Image], max_resolution: tuple) -> List[Image.Image]:
    """Resize frames to max resolution while maintaining aspect ratio."""
    max_w, max_h = max_resolution

    resized = []
    for frame in frames:
        w, h = frame.size

        # Calculate scaling factor
        scale = min(max_w / w, max_h / h)

        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = frame.resize((new_w, new_h), Image.LANCZOS)

        resized.append(frame)

    return resized


def annotate(
        image: Image.Image,
        detections: sv.Detections,
        label: Optional[str] = None
) -> Image.Image:
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
        text_scale=text_scale,
        text_padding=5,
        text_color=sv.Color.BLACK,
        text_thickness=1,
    )

    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(annotated_image, detections)
    annotated_image = box_annotator.annotate(annotated_image, detections)

    if label:
        labels = [
            f"id={track_id} | {label} {conf:.2f}"
            for track_id, conf in zip(detections.class_id, detections.confidence)
        ]
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)

    return annotated_image


def sam3_video_frame_to_detections(
        frame_outputs: Dict[str, Any],
        conf_thresh: float = 0.05
) -> Optional[sv.Detections]:
    """Convert SAM3 Video frame outputs to sv.Detections."""
    if frame_outputs is None:
        return None

    boxes = frame_outputs["boxes"]
    scores = frame_outputs["scores"]
    masks = frame_outputs["masks"]
    object_ids = frame_outputs["object_ids"]

    keep = scores > conf_thresh
    if keep.sum() == 0:
        return None

    boxes = boxes[keep]
    scores = scores[keep]
    masks = masks[keep]
    object_ids = object_ids[keep]

    boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
    scores_np = scores.detach().cpu().numpy().astype(np.float32)
    masks_np = masks.detach().cpu().numpy()
    obj_ids_np = object_ids.detach().cpu().numpy().astype(np.int32)

    if masks_np.dtype != bool:
        masks_bin = masks_np > 0.5
    else:
        masks_bin = masks_np

    detections = sv.Detections(
        xyxy=boxes_np,
        mask=masks_bin,
        confidence=scores_np,
        class_id=obj_ids_np,
    )

    return detections


def process_video_in_chunks(
        video_frames: List[Image.Image],
        model,
        processor,
        device,
        dtype,
        text_prompt: str,
        conf_thresh: float,
        chunk_size: int,
        overlap: int
) -> Dict[int, Dict[str, Any]]:
    """
    Process video in chunks to avoid OOM.

    Strategy: Process overlapping chunks, then merge results.
    """
    num_frames = len(video_frames)
    all_outputs = {}

    # Calculate chunks with overlap
    chunks = []
    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        chunks.append((start, end))
        start = end - overlap if end < num_frames else num_frames

    print(f"\nProcessing {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")

    for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
        print(f"\n--- Chunk {chunk_idx + 1}/{len(chunks)}: frames {start_idx}-{end_idx} ---")

        # Extract chunk frames
        chunk_frames = video_frames[start_idx:end_idx]

        # Clear GPU cache before each chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Initialize inference session for this chunk
        inference_session = processor.init_video_session(
            video=chunk_frames,
            inference_device=device,
            inference_state_device="cpu" if USE_CPU_OFFLOAD else device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=dtype
        )

        # Add text prompt
        inference_session = processor.add_text_prompt(
            inference_session=inference_session,
            text=text_prompt,
        )

        # Process chunk
        chunk_outputs = {}
        for model_outputs in model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=len(chunk_frames),
        ):
            processed_outputs = processor.postprocess_outputs(
                inference_session,
                model_outputs
            )
            # Map to global frame index
            global_frame_idx = start_idx + model_outputs.frame_idx
            chunk_outputs[global_frame_idx] = processed_outputs

        # Merge outputs (prioritize non-overlap regions)
        for frame_idx, outputs in chunk_outputs.items():
            # For overlap regions, keep the first occurrence
            if frame_idx not in all_outputs:
                all_outputs[frame_idx] = outputs

        print(f"Chunk {chunk_idx + 1} complete: processed {len(chunk_outputs)} frames")

        # Clean up
        del inference_session
        del chunk_outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return all_outputs


def main():
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle MKV conversion
    temp_mp4_path = None
    temp_dir = None

    if video_path.suffix.lower() == ".mkv":
        temp_dir = Path(tempfile.mkdtemp())
        temp_mp4_path = convert_mkv_to_mp4(video_path, temp_dir)
        processing_video_path = temp_mp4_path
    else:
        processing_video_path = video_path

    try:
        accelerator = Accelerator()
        device = accelerator.device
        print("Torch device:", device)

        # Choose dtype
        if device.type == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Load SAM3 Video model
        print("Loading SAM3 Video model...")
        model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)

        # Limit tracked objects to reduce memory
        model.max_num_objects = MAX_TRACKED_OBJECTS

        processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
        print("Model loaded.\n")

        # Load video frames
        video_frames, video_meta = load_video(str(processing_video_path))
        num_frames = len(video_frames)
        print(f"Loaded video with {num_frames} frames.")

        # Apply frame skipping if configured
        if FRAME_SKIP > 1:
            print(f"Applying frame skip: processing every {FRAME_SKIP} frame(s)")
            video_frames = video_frames[::FRAME_SKIP]
            print(f"Reduced to {len(video_frames)} frames")

        # Apply resolution reduction if configured
        if MAX_RESOLUTION is not None:
            print(f"Resizing frames to max resolution: {MAX_RESOLUTION}")
            video_frames = resize_frames(video_frames, MAX_RESOLUTION)

        fps = OUTPUT_FPS

        # Memory-efficient processing strategy
        print("\n" + "=" * 60)
        print("MEMORY OPTIMIZATION STRATEGY")
        print("=" * 60)
        print(f"Chunk size: {CHUNK_SIZE} frames")
        print(f"Overlap: {OVERLAP_FRAMES} frames")
        print(f"CPU offload: {USE_CPU_OFFLOAD}")
        print(f"Max tracked objects: {MAX_TRACKED_OBJECTS}")
        print(f"Frame skip: {FRAME_SKIP}")
        print(f"Resolution limit: {MAX_RESOLUTION or 'None (original)'}")
        print("=" * 60)

        # Process video in chunks
        outputs_per_frame = process_video_in_chunks(
            video_frames=video_frames,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            text_prompt=TEXT_PROMPT,
            conf_thresh=CONF_THRESH,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP_FRAMES
        )

        print(f"\nTotal processed frames: {len(outputs_per_frame)}\n")

        # Build annotated frames
        frames_to_write: List[np.ndarray] = []

        for idx, frame in enumerate(video_frames):
            if isinstance(frame, Image.Image):
                frame_pil = frame.convert("RGB")
            else:
                frame_pil = Image.fromarray(frame).convert("RGB")

            frame_outputs = outputs_per_frame.get(idx, None)

            if not frame_outputs:
                frames_to_write.append(np.array(frame_pil))
                continue

            detections = sam3_video_frame_to_detections(
                frame_outputs,
                conf_thresh=CONF_THRESH
            )

            if detections is None or len(detections) == 0:
                frames_to_write.append(np.array(frame_pil))
                continue

            annotated_pil = annotate(frame_pil, detections, label=TEXT_PROMPT)
            frames_to_write.append(np.array(annotated_pil))

        # Write annotated video
        print(f"Writing annotated video to: {out_path}")
        iio.imwrite(
            out_path,
            frames_to_write,
            fps=fps,
            codec="libx264",
        )
        print("✓ Done!")

    finally:
        # Cleanup temporary files
        if temp_mp4_path and temp_mp4_path.exists():
            print(f"\nCleaning up temporary file: {temp_mp4_path}")
            temp_mp4_path.unlink()

        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()