import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from accelerate import Accelerator
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from transformers.video_utils import load_video

import imageio.v3 as iio  # used for writing preview video


# ------------------ CONFIG ------------------
DATASET_ROOT = Path("/home/evox5090ia/datasets/smcb_2025-07-31-06-33-34_yolo")
SAM_ROOT = Path("/home/evox5090ia/sam3")
VIDEO_ID = "000015"  # <-- change this per video, e.g. "000078"

YOLO_OVERFLOW_CLASS_ID = 2  # your 'overflow' class ID

LABELS_DIR = DATASET_ROOT / "labels"
VIDEOS_DIR = DATASET_ROOT / "videos"
OUTPUT_MASK_DIR = SAM_ROOT / "sam3_overflow_masks"


# Write a preview video with masks overlaid
WRITE_PREVIEW_VIDEO = True
PREVIEW_VIDEO_PATH = SAM_ROOT / "sam3_overflow_previews" / f"{VIDEO_ID}_overflow_preview.mp4"
# ---------------------------------------------


def parse_frame_idx_from_name(stem: str, video_id: str) -> int:
    """
    Parse frame index from file stem: e.g. "000078_000261" -> 261
    Assumes pattern VIDEOID_FRAMEIDX.
    """
    if not stem.startswith(video_id + "_"):
        raise ValueError(f"Unexpected filename pattern for {stem}, expected '{video_id}_XXXXXX'")
    frame_str = stem.split("_", 1)[1]
    return int(frame_str)


def load_overflow_tracks_for_video(
    video_id: str,
    labels_dir: Path,
    img_width: int,
    img_height: int,
    overflow_class_id: int = YOLO_OVERFLOW_CLASS_ID,
) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
    """
    Load YOLO overflow tracks for a given video.

    Returns:
      tracks: dict[track_id] -> list of (frame_idx, (x1, y1, x2, y2)) in pixel coords
    """
    pattern = f"{video_id}_*.txt"
    label_paths = sorted(labels_dir.glob(pattern))

    tracks: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}

    for label_path in label_paths:
        frame_idx = parse_frame_idx_from_name(label_path.stem, video_id)

        with label_path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue

                cls_id = int(parts[0])
                track_id = int(parts[1])
                if cls_id != overflow_class_id:
                    continue

                xc = float(parts[2])
                yc = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                # YOLO normalized [0,1] -> pixel coords
                bw = w * img_width
                bh = h * img_height
                cx = xc * img_width
                cy = yc * img_height

                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0

                tracks.setdefault(track_id, []).append(
                    (frame_idx, (x1, y1, x2, y2))
                )

    # Sort each track's frames by frame_idx
    for tid in tracks:
        tracks[tid] = sorted(tracks[tid], key=lambda t: t[0])

    return tracks


def main():
    video_path = VIDEOS_DIR / f"{VIDEO_ID}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_MASK_DIR / VIDEO_ID).mkdir(parents=True, exist_ok=True)

    if WRITE_PREVIEW_VIDEO:
        PREVIEW_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading video...")
    video_frames, video_meta = load_video(str(video_path))
    num_frames = len(video_frames)
    print(f"Video: {num_frames} frames")

    # Get frame resolution from first frame
    first_frame = video_frames[0]
    if isinstance(first_frame, Image.Image):
        img_w, img_h = first_frame.size
    else:
        # numpy array HxWxC
        img_h, img_w = first_frame.shape[:2]
    print(f"Resolution: {img_w}x{img_h}")

    print("Loading YOLO overflow tracks...")
    tracks = load_overflow_tracks_for_video(
        video_id=VIDEO_ID,
        labels_dir=LABELS_DIR,
        img_width=img_w,
        img_height=img_h,
    )
    if not tracks:
        print("No overflow tracks found for this video.")
        return

    print(f"Found {len(tracks)} overflow tracks: {sorted(tracks.keys())}")

    # Setup SAM3 Tracker Video
    accelerator = Accelerator()
    device = accelerator.device
    print("Torch device:", device)

    if device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("Loading SAM3 Tracker Video...")
    model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
    processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    print("Model loaded.\n")

    # Optional: collect preview frames if you want a video with masks drawn
    preview_frames = [None] * num_frames if WRITE_PREVIEW_VIDEO else None

    # ---- Track each overflow object separately ----
    for track_idx, (track_id, frame_boxes) in enumerate(tracks.items(), start=1):
        print(f"\n=== Tracking overflow track {track_id} ({track_idx}/{len(tracks)}) ===")

        # New session per track
        session = processor.init_video_session(
            video=video_frames,
            inference_device=device,
            dtype=dtype,
        )

        # First frame where this object appears
        first_frame_idx, (x1, y1, x2, y2) = frame_boxes[0]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        print(f"  First frame for track {track_id}: {first_frame_idx}, click=({cx:.1f}, {cy:.1f})")

        # Build point prompt
        points = [[[[cx, cy]]]]      # [batch=1, obj=1, points=1, 2]
        labels = [[[1]]]             # positive click

        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=first_frame_idx,
            obj_ids=1,               # local object id in this session
            input_points=points,
            input_labels=labels,
        )

        # Map: frame_idx -> binary mask for THIS track
        masks_for_track: Dict[int, np.ndarray] = {}

        # Run propagation through the video, starting from the first frame where this track exists
        frame_indices_for_track = {f for f, _ in frame_boxes}

        for out in model.propagate_in_video_iterator(
            inference_session=session,
            start_frame_idx=first_frame_idx,
        ):
            f_idx = out.frame_idx

            # Only keep frames where this track exists in YOLO (optional)
            if f_idx not in frame_indices_for_track:
                continue

            # Post-process masks to original resolution
            video_res_masks = processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[[session.video_height, session.video_width]],
                binarize=False,
            )[0]  # [num_objects, 1, H, W] – here num_objects=1

            # First object, first mask
            mask_tensor = video_res_masks[0, 0]  # [H, W]
            mask_bin = (mask_tensor > 0.0).cpu().numpy().astype(np.uint8) * 255

            masks_for_track[f_idx] = mask_bin

        # ---- Save masks for this track to disk ----
        out_dir = OUTPUT_MASK_DIR / VIDEO_ID
        for frame_idx, mask in masks_for_track.items():
            out_path = out_dir / f"{VIDEO_ID}_{frame_idx:06d}_track{track_id}.png"
            Image.fromarray(mask).save(out_path)
        print(f"  Saved {len(masks_for_track)} masks for track {track_id} to {out_dir}")

        # ---- Optionally build preview frames with alpha overlay ----
        if WRITE_PREVIEW_VIDEO:
            for f_idx, mask in masks_for_track.items():
                # Base frame: if already has overlays from previous tracks, use that;
                # otherwise start from the raw video frame.
                if preview_frames[f_idx] is None:
                    frame = video_frames[f_idx]
                    if not isinstance(frame, Image.Image):
                        base_img = Image.fromarray(frame).convert("RGB")
                    else:
                        base_img = frame.convert("RGB")
                else:
                    base_img = Image.fromarray(preview_frames[f_idx]).convert("RGB")

                # Simple RGBA overlay: red mask with alpha from mask
                overlay_alpha = Image.fromarray(mask).convert("L")
                color = Image.new("RGBA", base_img.size, (255, 0, 0, 0))  # red base
                color.putalpha(overlay_alpha)

                base_rgba = base_img.convert("RGBA")
                composited = Image.alpha_composite(base_rgba, color).convert("RGB")

                preview_frames[f_idx] = np.array(composited)

    # ---- Write preview video if requested ----
    if WRITE_PREVIEW_VIDEO:
        # Fill any None frames with original frames
        for i in range(num_frames):
            if preview_frames[i] is None:
                frame = video_frames[i]
                if not isinstance(frame, Image.Image):
                    preview_frames[i] = frame
                else:
                    preview_frames[i] = np.array(frame.convert("RGB"))

        fps = video_meta.get("fps", 10)
        PREVIEW_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting preview video with masks to: {PREVIEW_VIDEO_PATH}")
        iio.imwrite(
            PREVIEW_VIDEO_PATH,
            preview_frames,
            fps=fps,
            codec="libx264",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
