import torch
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import tempfile
import shutil
import gc
import json
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import supervision as sv
import imageio.v3 as iio
import subprocess

from accelerate import Accelerator
from transformers import Sam3TrackerModel, Sam3TrackerProcessor
from transformers.video_utils import load_video

# CLIP for classification
from transformers import CLIPProcessor, CLIPModel


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    VIDEO_PATH = "/home/evox5090ia/Downloads/2025-12-27_13-43-36/video/traseira/2025-12-27_13-43-36_traseira_100.mkv"
    OUTPUT_PATH = "/home/evox5090ia/Downloads/output_CONTAMINATION_caixapapel.mp4"
    REPORT_PATH = "/home/evox5090ia/Downloads/contamination_report_caixapapel.json"

    # Collection Type - CHANGE THIS BASED ON YOUR COLLECTION
    COLLECTION_TYPE = "glass"  # Options: "paper", "plastic_metal", "glass", "organic"

    # Video settings
    OUTPUT_FPS = 10
    FRAME_SKIP = 2
    PROCESS_EVERY_N_FRAMES = 5
    CLEAR_MEMORY_EVERY_N_FRAMES = 20

    # Resolution settings
    PROCESSING_RESOLUTION = 960
    OUTPUT_ORIGINAL_RESOLUTION = True

    # ROI settings
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

    # Classification settings
    MIN_CLASSIFICATION_CONFIDENCE = 0.3  # Minimum confidence to classify
    CLASSIFY_EVERY_N_DETECTIONS = 1  # Classify every detection (set to 5 to classify every 5th)


# Waste classification categories (CLIP zero-shot)
WASTE_CLASSES = {
    "paper_cardboard": [
        "paper", "cardboard", "newspaper", "magazine",
        "office paper", "cardboard box", "paper bag"
    ],
    "plastic": [
        "plastic bottle", "plastic container", "plastic bag",
        "PET bottle", "plastic packaging", "plastic cup"
    ],
    "metal": [
        "metal can", "aluminum can", "tin can",
        "metal container", "beverage can"
    ],
    "glass": [
        "glass bottle", "glass jar", "glass container",
        "wine bottle", "beer bottle"
    ],
    "organic": [
        "food waste", "fruit", "vegetable", "organic waste",
        "biodegradable waste", "food scraps"
    ],
    "other": [
        "mixed waste", "general trash", "unidentifiable waste"
    ]
}

# Flatten for CLIP
ALL_CLASSES = []
CLASS_TO_CATEGORY = {}
for category, items in WASTE_CLASSES.items():
    for item in items:
        ALL_CLASSES.append(item)
        CLASS_TO_CATEGORY[item] = category

# Contamination rules: what's allowed for each collection type
COLLECTION_RULES = {
    "paper": {
        "allowed": ["paper_cardboard"],
        "name": "Paper & Cardboard Collection"
    },
    "plastic_metal": {
        "allowed": ["plastic", "metal"],
        "name": "Plastic & Metal Collection"
    },
    "glass": {
        "allowed": ["glass"],
        "name": "Glass Collection"
    },
    "organic": {
        "allowed": ["organic"],
        "name": "Organic Waste Collection"
    }
}

COLOR_PALETTE = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])


# ============================================================================
# CONTAMINATION DETECTOR CLASS
# ============================================================================

class ContaminationDetector:
    def __init__(self, collection_type: str):
        self.collection_type = collection_type
        self.rules = COLLECTION_RULES.get(collection_type, COLLECTION_RULES["paper"])
        self.allowed_categories = self.rules["allowed"]
        self.contamination_log = []

    def is_contamination(self, category: str) -> bool:
        """Check if detected category is contamination for this collection type."""
        return category not in self.allowed_categories and category != "other"

    def get_severity(self, category: str) -> str:
        """Determine severity of contamination."""
        # Critical contaminations that damage recycling process
        critical_map = {
            "paper": ["glass", "metal", "organic"],
            "plastic_metal": ["organic", "glass"],
            "glass": ["plastic", "metal", "organic", "paper_cardboard"],
            "organic": ["plastic", "metal", "glass", "paper_cardboard"]
        }

        critical = critical_map.get(self.collection_type, [])

        if category in critical:
            return "CRITICAL"
        elif category != "other":
            return "MEDIUM"
        else:
            return "LOW"

    def log_contamination(self, frame_idx: int, timestamp: float,
                          detected_class: str, category: str,
                          confidence: float, bbox: np.ndarray):
        """Log a contamination event."""
        severity = self.get_severity(category)

        contamination = {
            "frame": int(frame_idx),
            "timestamp": float(timestamp),
            "collection_type": self.collection_type,
            "detected_class": detected_class,
            "category": category,
            "confidence": float(confidence),
            "severity": severity,
            "bbox": bbox.tolist()
        }

        self.contamination_log.append(contamination)
        return contamination

    def get_summary(self) -> Dict:
        """Generate contamination summary statistics."""
        if not self.contamination_log:
            return {
                "total_contaminations": 0,
                "collection_type": self.collection_type
            }

        severities = [c["severity"] for c in self.contamination_log]
        categories = [c["category"] for c in self.contamination_log]

        from collections import Counter
        category_counts = Counter(categories)
        severity_counts = Counter(severities)

        return {
            "total_contaminations": len(self.contamination_log),
            "collection_type": self.collection_type,
            "by_severity": dict(severity_counts),
            "by_category": dict(category_counts),
            "first_contamination_time": self.contamination_log[0]["timestamp"],
            "last_contamination_time": self.contamination_log[-1]["timestamp"]
        }


# ============================================================================
# CLIP CLASSIFIER
# ============================================================================

class WasteClassifier:
    def __init__(self, device):
        print("Loading CLIP model for classification...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        self.classes = ALL_CLASSES
        print(f"Loaded CLIP with {len(self.classes)} waste classes\n")

    def classify(self, image_crop: Image.Image) -> Tuple[str, str, float]:
        """
        Classify a cropped object image.

        Returns:
            Tuple of (detected_class, category, confidence)
        """
        inputs = self.processor(
            text=self.classes,
            images=image_crop,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)[0]
        class_idx = probs.argmax().item()
        confidence = probs[class_idx].item()

        detected_class = self.classes[class_idx]
        category = CLASS_TO_CATEGORY[detected_class]

        return detected_class, category, confidence


# ============================================================================
# GEOMETRY UTILITIES (from original script)
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
    x1, y1, x2, y2 = bbox

    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    if not point_in_polygon(center, roi_polygon):
        return False

    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    corners_in = sum(point_in_polygon(corner, roi_polygon) for corner in corners)

    return corners_in >= int(4 * threshold)


# ============================================================================
# FILTERING FUNCTIONS (from original script)
# ============================================================================

def has_strong_edges(mask: np.ndarray, min_density: float = 0.12) -> bool:
    if mask.sum() == 0:
        return False

    mask_uint8 = (mask * 255).astype(np.uint8)
    edges = cv2.Canny(mask_uint8, 50, 150)

    edge_pixels = np.sum(edges > 0)
    mask_pixels = np.sum(mask > 0)
    edge_density = edge_pixels / mask_pixels if mask_pixels > 0 else 0

    return edge_density >= min_density


def check_aspect_ratio(bbox: np.ndarray, min_ratio: float = 0.2, max_ratio: float = 5.0) -> bool:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1

    if height == 0:
        return False

    aspect_ratio = width / height
    return min_ratio <= aspect_ratio <= max_ratio


def filter_detections_by_roi(detections: sv.Detections, roi_polygon: np.ndarray, threshold: float = 0.75) -> Optional[
    sv.Detections]:
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
    if detections is None or len(detections) == 0:
        return None

    keep_indices = []

    for i in range(len(detections)):
        bbox = detections.xyxy[i]
        mask = detections.mask[i] if detections.mask is not None else None
        confidence = detections.confidence[i] if detections.confidence is not None else 0.0

        if confidence < Config.MIN_IOU_SCORE:
            continue

        if not check_aspect_ratio(bbox, Config.MIN_ASPECT_RATIO, Config.MAX_ASPECT_RATIO):
            continue

        if mask is not None:
            mask_area = np.sum(mask)
            if mask_area > roi_area * Config.MAX_MASK_AREA_RATIO:
                continue

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
# VIDEO PROCESSING (from original script)
# ============================================================================

def convert_mkv_to_mp4(mkv_path: Path, temp_dir: Path) -> Path:
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
# MASK GENERATION (from original script)
# ============================================================================

def generate_masks_for_frame(frame_pil: Image.Image, model, processor, device, dtype,
                             roi_polygon: Optional[np.ndarray] = None) -> Optional[sv.Detections]:
    img_np = np.array(frame_pil)
    h, w = img_np.shape[:2]

    if roi_polygon is not None:
        roi_bbox = get_roi_bounding_box(roi_polygon)
        x_min, y_min, x_max, y_max = roi_bbox
    else:
        x_min, y_min, x_max, y_max = 0, 0, w, h

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

    for i in range(0, len(input_points), Config.POINTS_PER_BATCH):
        batch_points = input_points[i:i + Config.POINTS_PER_BATCH]

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

    if roi_polygon is not None:
        roi_area = get_roi_area(roi_polygon)
        detections = apply_advanced_filters(detections, roi_polygon, roi_area, Config.USE_EDGE_DETECTION)

    return detections


# ============================================================================
# ANNOTATION WITH CONTAMINATION ALERTS
# ============================================================================

def draw_roi_on_image(image: Image.Image, roi_polygon: np.ndarray, color: tuple, thickness: int) -> Image.Image:
    draw = ImageDraw.Draw(image)
    points = [tuple(pt) for pt in roi_polygon]
    draw.polygon(points, outline=color, width=thickness)
    return image


def annotate_with_classification(image: Image.Image, detections: sv.Detections,
                                 classifications: List[Dict],
                                 roi_polygon: Optional[np.ndarray] = None,
                                 draw_roi: bool = False) -> Image.Image:
    """Annotate image with detections and classification results."""
    img_np = np.array(image)

    if draw_roi and roi_polygon is not None:
        pts = roi_polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img_np, [pts], True, Config.ROI_COLOR, Config.ROI_THICKNESS)

    # Draw masks and boxes
    mask_annotator = sv.MaskAnnotator(color=COLOR_PALETTE, opacity=0.3)
    box_annotator = sv.BoxAnnotator(color=COLOR_PALETTE, thickness=2)

    img_np = mask_annotator.annotate(img_np, detections)
    img_np = box_annotator.annotate(img_np, detections)

    # Draw classification labels
    for i, (bbox, classification) in enumerate(zip(detections.xyxy, classifications)):
        x1, y1, x2, y2 = bbox.astype(int)

        # Prepare label text
        detected_class = classification.get('class', 'unknown')
        confidence = classification.get('confidence', 0.0)
        is_contamination = classification.get('is_contamination', False)

        if is_contamination:
            label = f"⚠ {detected_class} ({confidence:.2f})"
            bg_color = (0, 0, 255)  # Red for contamination
            text_color = (255, 255, 255)
        else:
            label = f"✓ {detected_class} ({confidence:.2f})"
            bg_color = (0, 255, 0)  # Green for correct
            text_color = (0, 0, 0)

        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_np, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), bg_color, -1)

        # Draw label text
        cv2.putText(img_np, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    return Image.fromarray(img_np)


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def clear_memory():
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
        print(f"Data type: {dtype}\n")

        # Load SAM3 model
        print("Loading SAM3 model...")
        model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
        processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

        # Load CLIP classifier
        classifier = WasteClassifier(device)

        # Initialize contamination detector
        contamination_detector = ContaminationDetector(Config.COLLECTION_TYPE)

        print(f"{'=' * 70}")
        print(f"CONTAMINATION DETECTION MODE")
        print(f"{'=' * 70}")
        print(f"Collection Type: {contamination_detector.rules['name']}")
        print(f"Allowed Materials: {', '.join(contamination_detector.allowed_categories)}")
        print(f"{'=' * 70}\n")

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
        last_classifications = None

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

                    # Classify each detection
                    classifications = []
                    contaminations_in_frame = 0

                    for det_idx in range(len(detections)):
                        bbox = detections.xyxy[det_idx].astype(int)
                        x1, y1, x2, y2 = bbox

                        # Crop object
                        object_crop = orig_frame_pil.crop((x1, y1, x2, y2))

                        # Skip very small crops
                        if object_crop.size[0] < 20 or object_crop.size[1] < 20:
                            classifications.append({
                                'class': 'too_small',
                                'category': 'other',
                                'confidence': 0.0,
                                'is_contamination': False
                            })
                            continue

                        # Classify
                        detected_class, category, confidence = classifier.classify(object_crop)

                        # Check if contamination
                        is_contamination = False
                        if confidence >= Config.MIN_CLASSIFICATION_CONFIDENCE:
                            is_contamination = contamination_detector.is_contamination(category)

                            if is_contamination:
                                contaminations_in_frame += 1
                                timestamp = idx / Config.OUTPUT_FPS
                                contamination_detector.log_contamination(
                                    idx, timestamp, detected_class, category, confidence, bbox
                                )

                        classifications.append({
                            'class': detected_class,
                            'category': category,
                            'confidence': confidence,
                            'is_contamination': is_contamination
                        })

                    last_detections = detections
                    last_classifications = classifications

                    print(f"→ {len(detections)} objects, {contaminations_in_frame} contaminations")
                else:
                    last_detections = None
                    last_classifications = None
                    print("→ 0 objects")

                if idx % Config.CLEAR_MEMORY_EVERY_N_FRAMES == 0:
                    clear_memory()
            else:
                detections = last_detections
                classifications = last_classifications

            # Annotate frame
            roi_for_annotation = roi_polygon_original if Config.OUTPUT_ORIGINAL_RESOLUTION else roi_polygon_absolute

            if detections is None or len(detections) == 0:
                if Config.DRAW_ROI and roi_for_annotation is not None:
                    orig_frame_pil = draw_roi_on_image(orig_frame_pil, roi_for_annotation, Config.ROI_COLOR,
                                                       Config.ROI_THICKNESS)
                frames_to_write.append(np.array(orig_frame_pil))
            else:
                annotated_pil = annotate_with_classification(
                    orig_frame_pil, detections, classifications,
                    roi_polygon=roi_for_annotation, draw_roi=Config.DRAW_ROI
                )
                frames_to_write.append(np.array(annotated_pil))

        # Generate summary
        summary = contamination_detector.get_summary()

        print(f"\n{'=' * 70}")
        print(f"CONTAMINATION DETECTION RESULTS")
        print(f"{'=' * 70}")
        print(f"Collection Type: {Config.COLLECTION_TYPE}")
        print(f"Total Contaminations: {summary['total_contaminations']}")

        if summary['total_contaminations'] > 0:
            print(f"\nBy Severity:")
            for severity, count in summary['by_severity'].items():
                print(f"  {severity}: {count}")

            print(f"\nBy Category:")
            for category, count in summary['by_category'].items():
                print(f"  {category}: {count}")

            print(f"\nFirst contamination at: {summary['first_contamination_time']:.1f}s")
            print(f"Last contamination at: {summary['last_contamination_time']:.1f}s")

        print(f"{'=' * 70}\n")

        # Save contamination report
        report_data = {
            "video_path": str(video_path),
            "collection_type": Config.COLLECTION_TYPE,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "contaminations": contamination_detector.contamination_log
        }

        with open(Config.REPORT_PATH, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"Contamination report saved to: {Config.REPORT_PATH}")

        # Write output video
        print(f"\nWriting annotated video to: {out_path}")
        iio.imwrite(out_path, frames_to_write, fps=Config.OUTPUT_FPS, codec="libx264")
        print("✅ Done!")

    finally:
        if temp_mp4_path and temp_mp4_path.exists():
            temp_mp4_path.unlink()
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()