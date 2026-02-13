import torch
from pathlib import Path
from typing import Optional, List, Tuple
import tempfile
import shutil
import gc

import numpy as np
from PIL import Image, ImageDraw
import cv2
import supervision as sv
import imageio.v3 as iio
import subprocess

from accelerate import Accelerator
from transformers import Sam3TrackerModel, Sam3TrackerProcessor
from transformers.video_utils import load_video


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    VIDEO_PATH = "/home/evox5090ia/Downloads/2025-12-23_08-56-33/video/traseira/2025-12-23_08-56-33_traseira_75.mkv"
    OUTPUT_PATH = "/home/evox5090ia/Downloads/output_FIXED_refact.mp4"

    # Video settings
    OUTPUT_FPS = 10
    FRAME_SKIP = 2
    PROCESS_EVERY_N_FRAMES = 5
    CLEAR_MEMORY_EVERY_N_FRAMES = 20

    # Resolution settings
    PROCESSING_RESOLUTION = 960
    OUTPUT_ORIGINAL_RESOLUTION = True

    # ROI settings (tighter polygon - trash zone only)
    USE_ROI = True
    ROI_POLYGON = [(0.30, 0.30), (0.70, 0.30), (0.70, 0.60), (0.30, 0.60)]
    DRAW_ROI = True
    ROI_COLOR = (0, 255, 0)
    ROI_THICKNESS = 3

    # SAM model settings
    CONF_THRESH = 0.25
    POINTS_PER_SIDE = 10
    POINTS_PER_BATCH = 12
    PRED_IOU_THRESH = 0.7
    STABILITY_SCORE_THRESH = 0.97
    MIN_MASK_REGION_AREA = 500

    # Advanced filtering
    MAX_MASK_AREA_RATIO = 0.10
    MIN_IOU_SCORE = 0.70
    USE_EDGE_DETECTION = True
    MIN_EDGE_DENSITY = 0.08
    MAX_OBJECTS_PER_FRAME = 15
    MIN_ASPECT_RATIO = 0.2
    MAX_ASPECT_RATIO = 5.0

    # Memory optimization
    USE_CPU_OFFLOAD = True


COLOR_PALETTE = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])


# ============================================================================
# GEOMETRY UTILITIES
# ============================================================================

def polygon_to_absolute(polygon: List[Tuple[float, float]], image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    return np.array([[int(x * width), int(y * height)] for x, y in polygon])


def get_roi_bounding_box(roi_polygon: np.ndarray) -> Tuple[int, int, int, int]:
    x_coords, y_coords = roi_polygon[:, 0], roi_polygon[:, 1]
    return int(np.min(x_coords)), int(np.min(y_coords)), int(np.max(x_coords)), int(np.max(y_coords))


def get_roi_area(roi_polygon: np.ndarray) -> float:
    x, y = roi_polygon[:, 0], roi_polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    inside = False
    p1x, p1y = polygon[0]

    for i in range(1, len(polygon) + 1):
        p2x, p2y = polygon[i % len(polygon)]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def bbox_in_roi(bbox: np.ndarray, roi_polygon: np.ndarray, threshold: float = 0.75) -> bool:
    """Check if bbox is in ROI. Requires center inside + 75% of corners inside."""
    x1, y1, x2, y2 = bbox

    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    if not point_in_polygon(center, roi_polygon):
        return False

    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    corners_in = sum(point_in_polygon(corner, roi_polygon) for corner in corners)

    return corners_in >= int(4 * threshold)


# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def has_strong_edges(mask: np.ndarray, min_density: float = 0.12) -> bool:
    """Detect real object boundaries vs texture patterns."""
    if mask.sum() == 0:
        return False

    mask_uint8 = (mask * 255).astype(np.uint8)
    edges = cv2.Canny(mask_uint8, 50, 150)

    edge_pixels = np.sum(edges > 0)
    mask_pixels = np.sum(mask > 0)
    edge_density = edge_pixels / mask_pixels if mask_pixels > 0 else 0

    return edge_density >= min_density


def check_aspect_ratio(bbox: np.ndarray, min_ratio: float = 0.2, max_ratio: float = 5.0) -> bool:
    """Filter elongated objects (container ribs or artifacts)."""
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1

    if height == 0:
        return False

    aspect_ratio = width / height
    return min_ratio <= aspect_ratio <= max_ratio


def filter_detections_by_roi(detections: sv.Detections, roi_polygon: np.ndarray, threshold: float = 0.75) -> Optional[
    sv.Detections]:
    """Filter detections to keep only those within ROI."""
    if detections is None or len(detections) == 0:
        return None

    keep_mask = np.array([bbox_in_roi(bbox, roi_polygon, threshold) for bbox in detections.xyxy])

    if not keep_mask.any():
        return None

    return sv.Detections(
        xyxy=detections.xyxy[keep_mask],
        mask=detections.mask[keep_mask] if detections.mask is not None else None,
        class_id=detections.class_id[keep_mask] if detections.class_id is not None else None,
        confidence=detections.confidence[keep_mask] if detections.confidence is not None else None,
        tracker_id=detections.tracker_id[keep_mask] if detections.tracker_id is not None else None
    )


def apply_advanced_filters(detections: sv.Detections, roi_polygon: np.ndarray, roi_area: float,
                           use_edge_detection: bool = True) -> Optional[sv.Detections]:
    """Apply all advanced filtering: size, edge detection, aspect ratio, IOU."""
    if detections is None or len(detections) == 0:
        return None

    keep_indices = []

    for i in range(len(detections)):
        bbox = detections.xyxy[i]
        mask = detections.mask[i] if detections.mask is not None else None
        confidence = detections.confidence[i] if detections.confidence is not None else 0.0

        # Filter by IOU score
        if confidence < Config.MIN_IOU_SCORE:
            continue

        # Filter by aspect ratio
        if not check_aspect_ratio(bbox, Config.MIN_ASPECT_RATIO, Config.MAX_ASPECT_RATIO):
            continue

        # Filter by size relative to ROI
        if mask is not None:
            mask_area = np.sum(mask)
            if mask_area > roi_area * Config.MAX_MASK_AREA_RATIO:
                continue

            # Filter by edge detection
            if use_edge_detection and not has_strong_edges(mask, Config.MIN_EDGE_DENSITY):
                continue

        keep_indices.append(i)

    if not keep_indices:
        return None

    keep_mask = np.zeros(len(detections), dtype=bool)
    keep_mask[keep_indices] = True

    return sv.Detections(
        xyxy=detections.xyxy[keep_mask],
        mask=detections.mask[keep_mask] if detections.mask is not None else None,
        class_id=detections.class_id[keep_mask] if detections.class_id is not None else None,
        confidence=detections.confidence[keep_mask] if detections.confidence is not None else None,
        tracker_id=detections.tracker_id[keep_mask] if detections.tracker_id is not None else None
    )


def limit_detections(detections: sv.Detections, max_objects: int) -> Optional[sv.Detections]:
    """Limit to top N detections by confidence."""
    if detections is None or len(detections) == 0:
        return None

    if len(detections) <= max_objects:
        return detections

    if detections.confidence is None:
        return sv.Detections(
            xyxy=detections.xyxy[:max_objects],
            mask=detections.mask[:max_objects] if detections.mask is not None else None,
            class_id=detections.class_id[:max_objects] if detections.class_id is not None else None,
            confidence=None,
            tracker_id=detections.tracker_id[:max_objects] if detections.tracker_id is not None else None
        )

    sorted_indices = np.argsort(detections.confidence)[::-1][:max_objects]

    return sv.Detections(
        xyxy=detections.xyxy[sorted_indices],
        mask=detections.mask[sorted_indices] if detections.mask is not None else None,
        class_id=detections.class_id[sorted_indices] if detections.class_id is not None else None,
        confidence=detections.confidence[sorted_indices],
        tracker_id=detections.tracker_id[sorted_indices] if detections.tracker_id is not None else None
    )


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def convert_mkv_to_mp4(mkv_path: Path, temp_dir: Path) -> Path:
    """Convert MKV to MP4 using FFmpeg."""
    mp4_path = temp_dir / f"{mkv_path.stem}.mp4"

    cmd = [
        "ffmpeg", "-i", str(mkv_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-y", str(mp4_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    return mp4_path


def resize_frames(frames: List, max_size: int) -> List[Image.Image]:
    """Resize frames to max dimension while maintaining aspect ratio."""
    resized = []

    for frame in frames:
        if isinstance(frame, Image.Image):
            img = frame
        else:
            img = Image.fromarray(frame)

        w, h = img.size
        if max(w, h) <= max_size:
            resized.append(img)
            continue

        if w > h:
            new_w, new_h = max_size, int(h * max_size / w)
        else:
            new_w, new_h = int(w * max_size / h), max_size

        resized.append(img.resize((new_w, new_h), Image.LANCZOS))

    return resized


def upscale_detections(detections: sv.Detections, target_size: Tuple[int, int],
                       source_size: Tuple[int, int]) -> sv.Detections:
    """Upscale detection coordinates and masks to target resolution."""
    scale_x = target_size[0] / source_size[0]
    scale_y = target_size[1] / source_size[1]

    scaled_xyxy = detections.xyxy.astype(np.float32).copy()
    scaled_xyxy[:, [0, 2]] *= scale_x
    scaled_xyxy[:, [1, 3]] *= scale_y
    scaled_xyxy = np.round(scaled_xyxy).astype(np.int32)

    scaled_mask = None
    if detections.mask is not None:
        scaled_mask = np.zeros((len(detections), target_size[1], target_size[0]), dtype=bool)
        for i, mask in enumerate(detections.mask):
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_uint8)
            scaled_mask_img = mask_img.resize(target_size, Image.LANCZOS)
            scaled_mask[i] = np.array(scaled_mask_img) > 127

    return sv.Detections(
        xyxy=scaled_xyxy,
        mask=scaled_mask,
        class_id=detections.class_id,
        confidence=detections.confidence,
        tracker_id=detections.tracker_id
    )


# ============================================================================
# MASK GENERATION
# ============================================================================

def generate_masks_for_frame(frame_pil: Image.Image, model, processor, device, dtype,
                             roi_polygon: Optional[np.ndarray] = None) -> Optional[sv.Detections]:
    """Generate masks for a single frame using SAM3 with point prompts."""
    img_np = np.array(frame_pil)
    h, w = img_np.shape[:2]

    # Generate point grid within ROI
    if roi_polygon is not None:
        roi_bbox = get_roi_bounding_box(roi_polygon)
        x_min, y_min, x_max, y_max = roi_bbox
    else:
        x_min, y_min, x_max, y_max = 0, 0, w, h

    # Create point grid
    x_step = (x_max - x_min) / (Config.POINTS_PER_SIDE + 1)
    y_step = (y_max - y_min) / (Config.POINTS_PER_SIDE + 1)

    input_points = []
    for i in range(1, Config.POINTS_PER_SIDE + 1):
        for j in range(1, Config.POINTS_PER_SIDE + 1):
            x = int(x_min + j * x_step)
            y = int(y_min + i * y_step)

            if roi_polygon is not None:
                if point_in_polygon(np.array([x, y]), roi_polygon):
                    input_points.append([x, y])
            else:
                input_points.append([x, y])

    if len(input_points) == 0:
        return None

    input_points = np.array(input_points)
    all_masks = []
    all_scores = []
    all_boxes = []

    # Process points in batches
    for i in range(0, len(input_points), Config.POINTS_PER_BATCH):
        batch_points = input_points[i:i + Config.POINTS_PER_BATCH]

        # Format for SAM3: [[[x, y]]] for each point
        formatted_points = [[[[int(p[0]), int(p[1])]]] for p in batch_points]
        formatted_labels = [[[1]] for _ in batch_points]

        inputs = processor(
            images=[frame_pil] * len(batch_points),
            input_points=formatted_points,
            input_labels=formatted_labels,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype=dtype)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu()
        )

        # Extract results
        for j, mask_set in enumerate(masks):
            if mask_set.shape[1] > 0:
                mask = mask_set[0, 0].numpy()

                y_indices, x_indices = np.where(mask > Config.CONF_THRESH)
                if len(y_indices) > 0:
                    x1, y1 = x_indices.min(), y_indices.min()
                    x2, y2 = x_indices.max(), y_indices.max()
                    area = (x2 - x1) * (y2 - y1)

                    if area >= Config.MIN_MASK_REGION_AREA:
                        all_masks.append(mask > Config.CONF_THRESH)
                        all_boxes.append([x1, y1, x2, y2])
                        score = outputs.iou_scores[j, 0].item() if hasattr(outputs, 'iou_scores') else 0.9
                        all_scores.append(score)

        del inputs, outputs, masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_masks) == 0:
        return None

    # Sort by score and limit
    masks_array = np.array(all_masks)
    scores_array = np.array(all_scores)
    boxes_array = np.array(all_boxes)

    sorted_indices = np.argsort(scores_array)[::-1][:Config.MAX_OBJECTS_PER_FRAME]

    masks_array = masks_array[sorted_indices]
    scores_array = scores_array[sorted_indices]
    boxes_array = boxes_array[sorted_indices]

    detections = sv.Detections(
        xyxy=boxes_array,
        mask=masks_array,
        confidence=scores_array,
        class_id=np.arange(len(masks_array)),
        tracker_id=np.arange(len(masks_array))
    )

    # Apply advanced filtering
    if roi_polygon is not None:
        roi_area = get_roi_area(roi_polygon)
        detections = apply_advanced_filters(detections, roi_polygon, roi_area, Config.USE_EDGE_DETECTION)

    return detections


# ============================================================================
# ANNOTATION
# ============================================================================

def draw_roi_on_image(image: Image.Image, roi_polygon: np.ndarray, color: tuple, thickness: int) -> Image.Image:
    """Draw ROI polygon on image."""
    draw = ImageDraw.Draw(image)
    points = [tuple(pt) for pt in roi_polygon]
    draw.polygon(points, outline=color, width=thickness)
    return image


def annotate(image: Image.Image, detections: sv.Detections, roi_polygon: Optional[np.ndarray] = None,
             draw_roi: bool = False) -> Image.Image:
    """Annotate image with detections and ROI."""
    img_np = np.array(image)

    if draw_roi and roi_polygon is not None:
        pts = roi_polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img_np, [pts], True, Config.ROI_COLOR, Config.ROI_THICKNESS)

    mask_annotator = sv.MaskAnnotator(color=COLOR_PALETTE, opacity=0.3)
    box_annotator = sv.BoxAnnotator(color=COLOR_PALETTE, thickness=2)
    label_annotator = sv.LabelAnnotator(color=COLOR_PALETTE, text_color=sv.Color.from_hex("#000000"))

    img_np = mask_annotator.annotate(img_np, detections)
    img_np = box_annotator.annotate(img_np, detections)

    labels = [f"#{i + 1}" for i in range(len(detections))]
    img_np = label_annotator.annotate(img_np, detections, labels)

    return Image.fromarray(img_np)


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# MAIN
# ============================================================================

def main():
    video_path = Path(Config.VIDEO_PATH)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_path = Path(Config.OUTPUT_PATH)
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
        # Setup PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        accelerator = Accelerator()
        device = accelerator.device
        print(f"Device: {device}")

        dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        print(f"Data type: {dtype}")

        # Load model
        print("\nLoading SAM3 model...")
        model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
        processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

        # Load video
        video_frames, video_meta = load_video(str(processing_video_path))
        print(f"Loaded {len(video_frames)} frames")

        # Get original resolution
        first_frame = video_frames[0]
        original_size = first_frame.size if isinstance(first_frame, Image.Image) else (first_frame.shape[1],
                                                                                       first_frame.shape[0])
        print(f"Original resolution: {original_size[0]}x{original_size[1]}")

        # Apply frame skip
        if Config.FRAME_SKIP > 1:
            video_frames = video_frames[::Config.FRAME_SKIP]
            print(f"Frame skip: {Config.FRAME_SKIP} → {len(video_frames)} frames")

        original_frames = video_frames if Config.OUTPUT_ORIGINAL_RESOLUTION else None

        # Resize for processing if needed
        processed_frames = video_frames
        processing_size = original_size

        if Config.PROCESSING_RESOLUTION and max(original_size) > Config.PROCESSING_RESOLUTION:
            processed_frames = resize_frames(video_frames, Config.PROCESSING_RESOLUTION)
            processing_size = processed_frames[0].size
            print(f"Processing resolution: {processing_size[0]}x{processing_size[1]}")

        # Setup ROI
        roi_polygon_absolute = None
        roi_polygon_original = None

        if Config.USE_ROI and Config.ROI_POLYGON:
            roi_polygon_absolute = polygon_to_absolute(Config.ROI_POLYGON, processing_size)
            roi_area = get_roi_area(roi_polygon_absolute)
            print(f"\nROI area: {roi_area:.0f} pixels²")

            if Config.OUTPUT_ORIGINAL_RESOLUTION:
                roi_polygon_original = polygon_to_absolute(Config.ROI_POLYGON, original_size)

        # Process frames
        print(f"\nProcessing frames (analyzing every {Config.PROCESS_EVERY_N_FRAMES})...\n")

        frames_to_write = []
        total_detections = 0
        frames_with_detections = 0
        last_detections = None

        for idx in range(len(processed_frames)):
            frame = processed_frames[idx]
            frame_pil = frame.convert("RGB") if isinstance(frame, Image.Image) else Image.fromarray(frame).convert(
                "RGB")

            # Get original resolution frame for output
            if Config.OUTPUT_ORIGINAL_RESOLUTION:
                orig_frame = original_frames[idx]
                orig_frame_pil = orig_frame.convert("RGB") if isinstance(orig_frame, Image.Image) else Image.fromarray(
                    orig_frame).convert("RGB")
            else:
                orig_frame_pil = frame_pil

            # Generate masks every N frames
            if idx % Config.PROCESS_EVERY_N_FRAMES == 0:
                progress = (idx / len(processed_frames)) * 100
                print(f"[{progress:5.1f}%] Frame {idx:4d}/{len(processed_frames)} ", end="")

                detections = generate_masks_for_frame(
                    frame_pil, model, processor, device, dtype,
                    roi_polygon=roi_polygon_absolute if Config.USE_ROI else None
                )

                if detections and len(detections) > 0:
                    total_detections += len(detections)
                    frames_with_detections += 1

                    # Upscale to original resolution if needed
                    if Config.OUTPUT_ORIGINAL_RESOLUTION and processing_size != original_size:
                        detections = upscale_detections(detections, original_size, processing_size)

                    last_detections = detections
                    print(f"→ {len(detections)} objects")
                else:
                    last_detections = None
                    print("→ 0 objects")

                if idx % Config.CLEAR_MEMORY_EVERY_N_FRAMES == 0:
                    clear_memory()
            else:
                detections = last_detections

            # Annotate frame
            roi_for_annotation = roi_polygon_original if Config.OUTPUT_ORIGINAL_RESOLUTION else roi_polygon_absolute

            if detections is None or len(detections) == 0:
                if Config.DRAW_ROI and roi_for_annotation is not None:
                    orig_frame_pil = draw_roi_on_image(orig_frame_pil, roi_for_annotation, Config.ROI_COLOR,
                                                       Config.ROI_THICKNESS)
                frames_to_write.append(np.array(orig_frame_pil))
            else:
                annotated_pil = annotate(orig_frame_pil, detections, roi_polygon=roi_for_annotation,
                                         draw_roi=Config.DRAW_ROI)
                frames_to_write.append(np.array(annotated_pil))

        # Print statistics
        processed_count = len(processed_frames) // Config.PROCESS_EVERY_N_FRAMES
        avg_objects = total_detections / frames_with_detections if frames_with_detections > 0 else 0

        print(f"\n{'=' * 60}")
        print(f"STATISTICS")
        print(f"{'=' * 60}")
        print(f"Frames analyzed: {processed_count}")
        print(f"Frames with objects: {frames_with_detections}")
        print(f"Total objects: {total_detections}")
        print(f"Average per frame: {avg_objects:.1f}")
        print(f"{'=' * 60}\n")

        # Write output video
        print(f"Writing video to: {out_path}")
        iio.imwrite(out_path, frames_to_write, fps=Config.OUTPUT_FPS, codec="libx264")
        print("✅ Done!")

    finally:
        if temp_mp4_path and temp_mp4_path.exists():
            temp_mp4_path.unlink()
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()