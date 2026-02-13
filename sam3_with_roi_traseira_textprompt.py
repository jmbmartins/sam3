import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import tempfile
import shutil
import gc

import numpy as np
from PIL import Image, ImageDraw

import supervision as sv

from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video

import imageio.v3 as iio  # pip install "imageio[ffmpeg]"
import subprocess

# --------- CONFIGURE THESE ---------
VIDEO_PATH = "/home/evox5090ia/Downloads/2025-12-23_08-56-33/video/traseira/2025-12-23_08-56-33_traseira_75.mkv"
OUTPUT_PATH = "/home/evox5090ia/Downloads/output_1.mp4"

CONF_THRESH = 0.05
OUTPUT_FPS = 10

# ===== ROI (Region of Interest) SETTINGS =====
# Define the detection zone as a polygon (relative coordinates 0.0 to 1.0)
USE_ROI = True

# ROI_POLYGON: Define as relative coordinates (0.0 to 1.0)
# Adjusted based on your container images
ROI_POLYGON = [
    (0.20, 0.03),  # Top-left
    (0.80, 0.03),  # Top-right
    (0.93, 0.70),  # Bottom-right (extended lower)
    (0.07, 0.70),  # Bottom-left (extended lower)
]

DRAW_ROI = True
ROI_COLOR = (255, 0, 0)  # Red
ROI_THICKNESS = 3

# ===== SEGMENTATION MODE =====
# Text prompt to detect objects - use generic terms for all objects
TEXT_PROMPT = "item"  # Options: "object", "thing", "item" - all will detect various objects

# ===== MEMORY OPTIMIZATION SETTINGS =====
CHUNK_SIZE = 50  # REDUCED from 100 to avoid OOM
OVERLAP_FRAMES = 5  # REDUCED from 10
MAX_RESOLUTION = 1280  # LIMIT resolution to reduce memory usage
FRAME_SKIP = 1
MAX_TRACKED_OBJECTS = 20  # REDUCED from 50 to limit memory
USE_CPU_OFFLOAD = True
ENABLE_MEMORY_EFFICIENT_MODE = True  # Process masks on CPU when possible
# ========================================


# Color palette for Supervision
COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])


def polygon_to_absolute(polygon: List[Tuple[float, float]], image_size: Tuple[int, int]) -> np.ndarray:
    """Convert relative polygon coordinates to absolute pixel coordinates."""
    width, height = image_size
    absolute_polygon = np.array([
        [int(x * width), int(y * height)]
        for x, y in polygon
    ])
    return absolute_polygon


def get_roi_bounding_box(roi_polygon: np.ndarray) -> Tuple[int, int, int, int]:
    """Get bounding box of ROI polygon."""
    x_coords = roi_polygon[:, 0]
    y_coords = roi_polygon[:, 1]
    return (
        int(np.min(x_coords)),
        int(np.min(y_coords)),
        int(np.max(x_coords)),
        int(np.max(y_coords))
    )


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y- p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def bbox_in_roi(bbox: np.ndarray, roi_polygon: np.ndarray, threshold: float = 0.3) -> bool:
    """Check if a bounding box is inside the ROI polygon."""
    x1, y1, x2, y2 = bbox

    # Get bbox center
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center = np.array([center_x, center_y])

    # Check if center is in polygon
    center_in = point_in_polygon(center, roi_polygon)

    if threshold >= 1.0:
        return center_in

    # Check corners
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ])

    corners_in = sum([point_in_polygon(corner, roi_polygon) for corner in corners])

    if center_in or corners_in >= int(4 * threshold):
        return True

    return False


def filter_detections_by_roi(
        detections: sv.Detections,
        roi_polygon: np.ndarray,
        threshold: float = 0.3
) -> Optional[sv.Detections]:
    """Filter detections to keep only those inside the ROI."""
    if detections is None or len(detections) == 0:
        return None

    keep_mask = np.array([
        bbox_in_roi(bbox, roi_polygon, threshold)
        for bbox in detections.xyxy
    ])

    if not keep_mask.any():
        return None

    filtered_detections = sv.Detections(
        xyxy=detections.xyxy[keep_mask],
        mask=detections.mask[keep_mask] if detections.mask is not None else None,
        confidence=detections.confidence[keep_mask] if detections.confidence is not None else None,
        class_id=detections.class_id[keep_mask] if detections.class_id is not None else None,
    )

    return filtered_detections


def draw_roi_on_image(
        image: Image.Image,
        roi_polygon: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int
) -> Image.Image:
    """Draw ROI polygon on image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    polygon_points = [tuple(point) for point in roi_polygon]
    draw.polygon(polygon_points, outline=color, width=thickness)

    return img_copy


def convert_mkv_to_mp4(mkv_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Convert MKV to MP4 using ffmpeg."""
    print(f"Converting MKV to MP4: {mkv_path}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())

    temp_mp4 = output_dir / f"{mkv_path.stem}_temp.mp4"

    cmd = [
        "ffmpeg",
        "-i", str(mkv_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-y",
        str(temp_mp4)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Conversion complete: {temp_mp4}\n")
        return temp_mp4
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")


def resize_frames(frames: List, max_size: int) -> List:
    """Resize frames to max dimension while maintaining aspect ratio."""
    resized = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            img = frame
        else:
            img = Image.fromarray(frame)

        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        resized.append(img)

    return resized


def sam3_video_frame_to_detections(
        frame_outputs: Dict[str, Any],
        conf_thresh: float = 0.0
) -> Optional[sv.Detections]:
    """Convert SAM3 Video frame outputs to supervision Detections."""
    if not frame_outputs:
        return None

    object_ids = frame_outputs.get("object_ids")
    masks = frame_outputs.get("masks")
    boxes = frame_outputs.get("boxes")
    scores = frame_outputs.get("scores")

    if object_ids is None or len(object_ids) == 0:
        return None

    if isinstance(object_ids, torch.Tensor):
        object_ids = object_ids.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # Filter by confidence threshold
    if scores is not None and conf_thresh > 0:
        valid_mask = scores >= conf_thresh
        if not valid_mask.any():
            return None

        object_ids = object_ids[valid_mask]
        masks = masks[valid_mask]
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    # Ensure masks are boolean
    if masks.dtype != bool:
        masks = masks > 0.5

    detections = sv.Detections(
        xyxy=boxes,
        mask=masks,
        confidence=scores,
        class_id=object_ids,
    )

    return detections


def annotate(
        image: Image.Image,
        detections: sv.Detections,
        label: str = "object",
        roi_polygon: Optional[np.ndarray] = None,
        draw_roi: bool = False
) -> Image.Image:
    """Annotate image with detections and ROI."""
    # Convert to numpy for supervision
    img_np = np.array(image)

    # Draw ROI first if enabled
    if draw_roi and roi_polygon is not None:
        image_with_roi = draw_roi_on_image(image, roi_polygon, ROI_COLOR, ROI_THICKNESS)
        img_np = np.array(image_with_roi)

    # Annotate with masks
    mask_annotator = sv.MaskAnnotator(color=COLOR)
    img_np = mask_annotator.annotate(scene=img_np, detections=detections)

    # Annotate with boxes
    box_annotator = sv.BoxAnnotator(color=COLOR)
    img_np = box_annotator.annotate(scene=img_np, detections=detections)

    # Add labels with object IDs
    labels = [f"{label} #{int(class_id)}" for class_id in detections.class_id]
    label_annotator = sv.LabelAnnotator(color=COLOR, text_color=sv.Color.BLACK)
    img_np = label_annotator.annotate(scene=img_np, detections=detections, labels=labels)

    return Image.fromarray(img_np)


def clear_memory():
    """Aggressively clear memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def process_video_in_chunks(
        video_frames: List,
        model: Sam3VideoModel,
        processor: Sam3VideoProcessor,
        device: torch.device,
        dtype: torch.dtype,
        conf_thresh: float,
        chunk_size: int = 50,
        overlap: int = 5,
        roi_polygon_absolute: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Process video in overlapping chunks to manage memory.
    """
    num_frames = len(video_frames)
    all_outputs = {}

    # Calculate chunks with overlap
    chunks = []
    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        chunks.append((start, end))
        if end >= num_frames:
            break
        start = end - overlap

    print(f"\nProcessing {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})\n")

    for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
        print(f"--- Chunk {chunk_idx + 1}/{len(chunks)}: frames {start_idx}-{end_idx} ---")

        # Clear memory before each chunk
        clear_memory()

        chunk_frames = video_frames[start_idx:end_idx]

        # Initialize inference session for this chunk
        inference_session = processor.init_video_session(
            video=chunk_frames,
            inference_device=device,
            processing_device="cpu",  # Force CPU processing to save GPU memory
            video_storage_device="cpu",  # Store video on CPU
            dtype=dtype,
        )

        # Add text prompt to detect all objects
        print(f"  Adding text prompt: '{TEXT_PROMPT}'")
        inference_session = processor.add_text_prompt(
            inference_session=inference_session,
            text=TEXT_PROMPT,
        )

        # Process chunk with memory management
        chunk_outputs = {}
        try:
            for frame_count, model_outputs in enumerate(model.propagate_in_video_iterator(
                    inference_session=inference_session,
                    max_frame_num_to_track=len(chunk_frames),
            )):
                # Move outputs to CPU immediately to free GPU memory
                with torch.no_grad():
                    processed_outputs = processor.postprocess_outputs(
                        inference_session,
                        model_outputs
                    )

                    # Convert tensors to CPU/numpy immediately
                    if processed_outputs.get("object_ids") is not None:
                        if isinstance(processed_outputs["object_ids"], torch.Tensor):
                            processed_outputs["object_ids"] = processed_outputs["object_ids"].cpu()
                    if processed_outputs.get("masks") is not None:
                        if isinstance(processed_outputs["masks"], torch.Tensor):
                            processed_outputs["masks"] = processed_outputs["masks"].cpu()
                    if processed_outputs.get("boxes") is not None:
                        if isinstance(processed_outputs["boxes"], torch.Tensor):
                            processed_outputs["boxes"] = processed_outputs["boxes"].cpu()
                    if processed_outputs.get("scores") is not None:
                        if isinstance(processed_outputs["scores"], torch.Tensor):
                            processed_outputs["scores"] = processed_outputs["scores"].cpu()

                global_frame_idx = start_idx + model_outputs.frame_idx
                chunk_outputs[global_frame_idx] = processed_outputs

                # Clear GPU memory every 10 frames
                if (frame_count + 1) % 10 == 0:
                    clear_memory()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ⚠ Memory error in chunk {chunk_idx + 1}, clearing and continuing...")
                clear_memory()
                # Save what we have so far
                for frame_idx, outputs in chunk_outputs.items():
                    if frame_idx not in all_outputs:
                        all_outputs[frame_idx] = outputs

                del inference_session
                del chunk_outputs
                clear_memory()
                continue
            else:
                raise

        for frame_idx, outputs in chunk_outputs.items():
            if frame_idx not in all_outputs:
                all_outputs[frame_idx] = outputs

        print(f"  Chunk {chunk_idx + 1} complete: processed {len(chunk_outputs)} frames")

        # Clean up
        del inference_session
        del chunk_outputs
        clear_memory()

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
        # Set PyTorch memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Enable memory efficient allocation
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        accelerator = Accelerator()
        device = accelerator.device
        print("Torch device:", device)

        if device.type == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        print("Loading SAM3 Video model...")
        model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
        model.max_num_objects = MAX_TRACKED_OBJECTS
        processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
        print("Model loaded.\n")

        # Load video frames
        video_frames, video_meta = load_video(str(processing_video_path))
        num_frames = len(video_frames)
        print(f"Loaded video with {num_frames} frames.")

        # Get image size from first frame
        first_frame = video_frames[0]
        if isinstance(first_frame, Image.Image):
            image_size = first_frame.size
        else:
            h, w = first_frame.shape[:2]
            image_size = (w, h)

        # Convert ROI to absolute coordinates
        roi_polygon_absolute = None
        if USE_ROI and ROI_POLYGON is not None:
            roi_polygon_absolute = polygon_to_absolute(ROI_POLYGON, image_size)
            print(f"\nROI enabled: {len(ROI_POLYGON)} vertices")
            print(f"ROI absolute coordinates: {roi_polygon_absolute.tolist()}")

        if FRAME_SKIP > 1:
            print(f"Applying frame skip: processing every {FRAME_SKIP} frame(s)")
            video_frames = video_frames[::FRAME_SKIP]
            print(f"Reduced to {len(video_frames)} frames")

        # Apply resolution limit
        if MAX_RESOLUTION is not None:
            print(f"Resizing frames to max resolution: {MAX_RESOLUTION}")
            video_frames = resize_frames(video_frames, MAX_RESOLUTION)
            if USE_ROI and ROI_POLYGON is not None:
                new_size = video_frames[0].size
                roi_polygon_absolute = polygon_to_absolute(ROI_POLYGON, new_size)
                print(f"ROI updated for new resolution: {roi_polygon_absolute.tolist()}")

        fps = OUTPUT_FPS

        print("\n" + "=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Segmentation mode: ALL OBJECTS (text prompt)")
        print(f"Text prompt: '{TEXT_PROMPT}'")
        print(f"Chunk size: {CHUNK_SIZE} frames")
        print(f"Overlap: {OVERLAP_FRAMES} frames")
        print(f"CPU offload: {USE_CPU_OFFLOAD}")
        print(f"Max tracked objects: {MAX_TRACKED_OBJECTS}")
        print(f"Frame skip: {FRAME_SKIP}")
        print(f"Resolution limit: {MAX_RESOLUTION or 'None (original)'}")
        print(f"ROI filtering: {USE_ROI}")
        print(f"Draw ROI: {DRAW_ROI}")
        print(f"Memory efficient mode: {ENABLE_MEMORY_EFFICIENT_MODE}")
        print("=" * 60)

        # Process video in chunks
        outputs_per_frame = process_video_in_chunks(
            video_frames=video_frames,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            conf_thresh=CONF_THRESH,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP_FRAMES,
            roi_polygon_absolute=roi_polygon_absolute,
        )

        print(f"\nTotal processed frames: {len(outputs_per_frame)}\n")

        # Build annotated frames
        frames_to_write: List[np.ndarray] = []
        total_detections_before_roi = 0
        total_detections_after_roi = 0

        for idx, frame in enumerate(video_frames):
            if isinstance(frame, Image.Image):
                frame_pil = frame.convert("RGB")
            else:
                frame_pil = Image.fromarray(frame).convert("RGB")

            frame_outputs = outputs_per_frame.get(idx, None)

            if not frame_outputs:
                if DRAW_ROI and roi_polygon_absolute is not None:
                    frame_pil = draw_roi_on_image(frame_pil, roi_polygon_absolute, ROI_COLOR, ROI_THICKNESS)
                frames_to_write.append(np.array(frame_pil))
                continue

            detections = sam3_video_frame_to_detections(
                frame_outputs,
                conf_thresh=CONF_THRESH
            )

            if detections is None or len(detections) == 0:
                if DRAW_ROI and roi_polygon_absolute is not None:
                    frame_pil = draw_roi_on_image(frame_pil, roi_polygon_absolute, ROI_COLOR, ROI_THICKNESS)
                frames_to_write.append(np.array(frame_pil))
                continue

            total_detections_before_roi += len(detections)

            # Apply ROI filtering
            if USE_ROI and roi_polygon_absolute is not None:
                detections = filter_detections_by_roi(
                    detections,
                    roi_polygon_absolute,
                    threshold=0.2  # More permissive for all-object mode
                )

            if detections is None or len(detections) == 0:
                if DRAW_ROI and roi_polygon_absolute is not None:
                    frame_pil = draw_roi_on_image(frame_pil, roi_polygon_absolute, ROI_COLOR, ROI_THICKNESS)
                frames_to_write.append(np.array(frame_pil))
                continue

            total_detections_after_roi += len(detections)

            annotated_pil = annotate(
                frame_pil,
                detections,
                label="object",
                roi_polygon=roi_polygon_absolute,
                draw_roi=DRAW_ROI
            )
            frames_to_write.append(np.array(annotated_pil))

        # Print statistics
        if USE_ROI:
            filtered_out = total_detections_before_roi - total_detections_after_roi
            print(f"\n" + "=" * 60)
            print("STATISTICS")
            print("=" * 60)
            print(f"Segmentation mode: ALL OBJECTS (text: '{TEXT_PROMPT}')")
            print(f"Total detections before ROI: {total_detections_before_roi}")
            print(f"Total detections after ROI:  {total_detections_after_roi}")
            print(f"Filtered out by ROI:         {filtered_out}")
            if total_detections_before_roi > 0:
                pct = (filtered_out / total_detections_before_roi) * 100
                print(f"Percentage filtered:         {pct:.1f}%")
            print("=" * 60 + "\n")

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
        if temp_mp4_path and temp_mp4_path.exists():
            print(f"\nCleaning up temporary file: {temp_mp4_path}")
            temp_mp4_path.unlink()

        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()