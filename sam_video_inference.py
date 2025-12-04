import torch
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image

import supervision as sv

from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.masking_utils import sliding_window_overlay
from transformers.video_utils import load_video

import imageio.v3 as iio  # pip install "imageio[ffmpeg]"


# --------- CONFIGURE THESE ---------
VIDEO_PATH = "/home/evox5090ia/datasets/smcb_2025-07-31-06-33-34_yolo/videos/000014.mp4"
TEXT_PROMPT = "trash left near container"
OUTPUT_PATH = "/home/evox5090ia/sam3/sam3_outputs/000014_trash_left_near_container.mp4"

CONF_THRESH = 0.05
# Use your real video FPS if you know it; otherwise 25 or 30 are usually OK.
OUTPUT_FPS = 10
# -----------------------------------


# Color palette for Supervision (same as your image script)
COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])


def annotate(
    image: Image.Image,
    detections: sv.Detections,
    label: Optional[str] = None
) -> Image.Image:
    # Compute a text scale adapted to resolution
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
        # Show track id + prompt + confidence
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
    """
    Convert one frame of SAM3 Video outputs into sv.Detections.

    frame_outputs is what you get from:
        processed_outputs = processor.postprocess_outputs(...)
    which contains (per HF docs):
        - object_ids: (N,)
        - scores:     (N,)
        - boxes:      (N, 4)   in XYXY pixel coords
        - masks:      (N, H, W) bool/float at original resolution
    """
    if frame_outputs is None:
        return None

    boxes = frame_outputs["boxes"]            # torch.Tensor [N, 4]
    scores = frame_outputs["scores"]          # torch.Tensor [N]
    masks = frame_outputs["masks"]            # torch.Tensor [N, H, W]
    object_ids = frame_outputs["object_ids"]  # torch.Tensor [N]

    # Filter by confidence directly in torch
    keep = scores > conf_thresh
    if keep.sum() == 0:
        return None

    boxes = boxes[keep]
    scores = scores[keep]
    masks = masks[keep]
    object_ids = object_ids[keep]

    # Convert to numpy
    boxes_np = boxes.detach().cpu().numpy().astype(np.float32)       # (M, 4)
    scores_np = scores.detach().cpu().numpy().astype(np.float32)     # (M,)
    masks_np = masks.detach().cpu().numpy()                          # (M, H, W)
    obj_ids_np = object_ids.detach().cpu().numpy().astype(np.int32)  # (M,)

    # Binarize masks if needed
    if masks_np.dtype != bool:
        masks_bin = masks_np > 0.5
    else:
        masks_bin = masks_np

    # IMPORTANT: use object_ids as class_id so same object keeps same color across frames
    class_ids = obj_ids_np

    detections = sv.Detections(
        xyxy=boxes_np,
        mask=masks_bin,
        confidence=scores_np,
        class_id=class_ids,
    )

    return detections


def main():
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator()
    device = accelerator.device
    print("Torch device:", device)

    # Choose dtype: bf16 on GPU, float32 otherwise
    if device.type == "cuda":
        # if your GPU doesn't support bf16, change to torch.float16
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load SAM3 Video model and processor (🤗 Transformers)
    print("Loading SAM3 Video model (may download weights the first time)...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)


    # model.max_num_objects = 45  # or 64–256 range


    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    print("Model loaded.\n")

    # Load video frames
    # load_video returns (frames, metadata). We only need frames here.
    video_frames, video_meta = load_video(str(video_path))
    num_frames = len(video_frames)
    print(f"Loaded video with {num_frames} frames.")

    # If you want real FPS from metadata (if available), override OUTPUT_FPS:
    # fps = video_meta.get("fps", OUTPUT_FPS)
    fps = OUTPUT_FPS

    # Initialize video inference session (pre-loaded video inference)
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        # inference_state_device="cpu", # TO MITIGATE OUT OF MEMORY
        processing_device="cpu",      # you can set this to device if you want
        video_storage_device="cpu",   # frames stored on CPU
        dtype=dtype
    )

    # Add text prompt
    print(f"Running text prompt: '{TEXT_PROMPT}'")
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=TEXT_PROMPT,
    )

    # Run propagation across the whole video
    outputs_per_frame: Dict[int, Dict[str, Any]] = {}

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=num_frames,
    ):
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs

    print(f"Processed {len(outputs_per_frame)} frames via SAM3 Video.\n")

    # Build annotated frames
    frames_to_write: List[np.ndarray] = []

    for idx, frame in enumerate(video_frames):
        # Ensure frame is a PIL image
        if isinstance(frame, Image.Image):
            frame_pil = frame.convert("RGB")
        else:
            frame_pil = Image.fromarray(frame).convert("RGB")

        frame_outputs = outputs_per_frame.get(idx, None)

        if not frame_outputs:
            # No outputs for this frame → just keep original
            frames_to_write.append(np.array(frame_pil))
            continue

        detections = sam3_video_frame_to_detections(frame_outputs, conf_thresh=CONF_THRESH)

        if detections is None or len(detections) == 0:
            # Nothing above threshold
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
    print("Done.")


if __name__ == "__main__":
    main()
