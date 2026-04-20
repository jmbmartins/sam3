import argparse
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video


# --------- CONFIGURE THESE ---------
VIDEO_PATH = "/home/evox5090ia/Downloads/helmet_detection/2026-03-03_15-25-16_traseira_3.mkv"
OUTPUT_PATH = "/home/evox5090ia/Downloads/helmet_detection/2026-03-03_15-25-16_traseira_3_1.mp4"

OPERATOR_PROMPT = "operator"
HELMET_PROMPT = "helmet"
FACE_PROMPT = "face"

OUTPUT_FPS = 10
CHUNK_SIZE = 80
OVERLAP_FRAMES = 8
FRAME_SKIP = 1
MAX_RESOLUTION = None  # Example: (1280, 720)
USE_CPU_OFFLOAD = True

OPERATOR_CONF = 0.50
HELMET_CONF = 0.35
FACE_CONF = 0.45

TRACK_IOU_THRESH = 0.30
STATE_WINDOW = 8
NO_HELMET_MIN_FRAMES = 2
HELMET_HOLD_FRAMES = 2
RECENT_HELMET_VETO_FRAMES = 2
MAX_TRACK_MISSES = 20

UPPER_BODY_RATIO = 0.30
FACE_MARGIN_RATIO = 0.05
OVERSIZED_BOX_AREA_RATIO = 0.22
OVERSIZED_BOX_WIDTH_RATIO = 0.55
OVERSIZED_BOX_HEIGHT_RATIO = 0.28
HEAD_ZONE_RATIO = 0.18
FACE_MAX_AREA_RATIO = 0.03
FACE_MAX_WIDTH_RATIO = 0.18
FACE_MAX_HEIGHT_RATIO = 0.18
HELMET_MAX_AREA_RATIO = 0.02
HELMET_MAX_WIDTH_RATIO = 0.16
HELMET_MAX_HEIGHT_RATIO = 0.16
FACE_HEADROOM_RATIO = 0.02
STRONG_FACE_CENTER_Y_MAX_RATIO = 0.40
STRONG_FACE_WIDTH_MIN_RATIO = 0.03
STRONG_FACE_HEIGHT_MIN_RATIO = 0.03
LABEL_FONT_SIZE = 22
LABEL_PADDING_X = 6
LABEL_PADDING_Y = 3
LABEL_MARGIN = 2
# -----------------------------------


@dataclass
class DetectionRecord:
    box: np.ndarray
    score: float


@dataclass
class OperatorDecision:
    raw_state: str
    reason: str
    operator_score: float
    face_records: List[DetectionRecord]
    helmet_records: List[DetectionRecord]
    oversized_operator: bool
    diagnostics: List[str]


def convert_mkv_to_mp4(mkv_path: Path, output_dir: Optional[Path] = None) -> Path:
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
        str(temp_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return temp_mp4


def resize_frames(frames: List[Image.Image], max_resolution: Tuple[int, int]) -> List[Image.Image]:
    max_w, max_h = max_resolution
    resized = []
    for frame in frames:
        w, h = frame.size
        scale = min(max_w / w, max_h / h)
        if scale < 1.0:
            frame = frame.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        resized.append(frame)
    return resized


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))

    return inter / (area_a + area_b - inter + 1e-6)


def center_of(box: np.ndarray) -> Tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def point_inside_box(point: Tuple[float, float], box: np.ndarray, margin_ratio: float = 0.0) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = x2 - x1
    h = y2 - y1
    margin_x = w * margin_ratio
    margin_y = h * margin_ratio
    px, py = point
    return (x1 + margin_x) <= px <= (x2 - margin_x) and (y1 + margin_y) <= py <= (y2 - margin_y)


def upper_body_region(box: np.ndarray, ratio: float = UPPER_BODY_RATIO) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    h = y2 - y1
    return np.array([x1, y1, x2, y1 + h * ratio], dtype=np.float32)


def head_zone_region(box: np.ndarray, ratio: float = HEAD_ZONE_RATIO) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    h = y2 - y1
    return np.array([x1, y1, x2, y1 + h * ratio], dtype=np.float32)


def frame_outputs_to_records(frame_outputs: Optional[Dict[str, Any]], conf_thresh: float) -> List[DetectionRecord]:
    if frame_outputs is None:
        return []

    boxes = frame_outputs["boxes"]
    scores = frame_outputs["scores"]
    keep = scores > conf_thresh
    if keep.sum() == 0:
        return []

    boxes_np = boxes[keep].detach().cpu().numpy().astype(np.float32)
    scores_np = scores[keep].detach().cpu().numpy().astype(np.float32)
    return [DetectionRecord(box=box, score=float(score)) for box, score in zip(boxes_np, scores_np)]


class TrackManager:
    def __init__(self) -> None:
        self.next_track_id = 0
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.state_history: Dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=STATE_WINDOW))

    def update(self, operator_records: List[DetectionRecord]) -> List[Tuple[int, DetectionRecord]]:
        assignments: List[Tuple[int, DetectionRecord]] = []
        matched = set()

        for record in operator_records:
            best_track_id = None
            best_iou = 0.0

            for track_id, track in self.tracks.items():
                iou = compute_iou(record.box, track["box"])
                if iou > best_iou and iou >= TRACK_IOU_THRESH:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self.next_track_id
                self.next_track_id += 1

            self.tracks[best_track_id] = {"box": record.box.copy(), "misses": 0}
            matched.add(best_track_id)
            assignments.append((best_track_id, record))

        for track_id in list(self.tracks.keys()):
            if track_id not in matched:
                self.tracks[track_id]["misses"] += 1
                if self.tracks[track_id]["misses"] > MAX_TRACK_MISSES:
                    self.tracks.pop(track_id, None)
                    self.state_history.pop(track_id, None)

        return assignments

    def smoothed_state(self, track_id: int, frame_state: str) -> str:
        history = self.state_history[track_id]
        history.append(frame_state)

        recent = list(history)
        if "helmet" in recent[-HELMET_HOLD_FRAMES:]:
            return "helmet"

        if frame_state == "no_helmet" and "helmet" in recent[-RECENT_HELMET_VETO_FRAMES:]:
            return "unknown"

        consecutive_no_helmet = 0
        for state in reversed(recent):
            if state == "no_helmet":
                consecutive_no_helmet += 1
            else:
                break

        if consecutive_no_helmet >= NO_HELMET_MIN_FRAMES:
            return "no_helmet"
        return "unknown"


def run_prompt_on_chunk(
    chunk_frames: List[Image.Image],
    prompt: str,
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, Dict[str, Any]]:
    inference_session = processor.init_video_session(
        video=chunk_frames,
        inference_device=device,
        inference_state_device="cpu" if USE_CPU_OFFLOAD else device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=prompt,
    )

    outputs: Dict[int, Dict[str, Any]] = {}
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=len(chunk_frames),
    ):
        outputs[model_outputs.frame_idx] = processor.postprocess_outputs(
            inference_session,
            model_outputs,
        )
    return outputs


def process_video_in_chunks(
    video_frames: List[Image.Image],
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    num_frames = len(video_frames)
    all_outputs: Dict[str, Dict[int, Dict[str, Any]]] = {
        "operator": {},
        "helmet": {},
        "face": {},
    }

    chunks: List[Tuple[int, int]] = []
    start = 0
    while start < num_frames:
        end = min(start + CHUNK_SIZE, num_frames)
        chunks.append((start, end))
        start = end - OVERLAP_FRAMES if end < num_frames else num_frames

    prompts = {
        "operator": OPERATOR_PROMPT,
        "helmet": HELMET_PROMPT,
        "face": FACE_PROMPT,
    }
    print(f"Processing {len(chunks)} chunks with prompts: {prompts}")

    for chunk_index, (start_idx, end_idx) in enumerate(chunks, start=1):
        chunk_frames = video_frames[start_idx:end_idx]
        print(f"Chunk {chunk_index}/{len(chunks)} frames {start_idx}-{end_idx - 1}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        for key, prompt in prompts.items():
            chunk_outputs = run_prompt_on_chunk(
                chunk_frames=chunk_frames,
                prompt=prompt,
                model=model,
                processor=processor,
                device=device,
                dtype=dtype,
            )

            for local_idx, outputs in chunk_outputs.items():
                global_idx = start_idx + local_idx
                if global_idx not in all_outputs[key]:
                    all_outputs[key][global_idx] = outputs

    return all_outputs


def operator_box_is_oversized(operator_box: np.ndarray, image_size: Tuple[int, int]) -> bool:
    width, height = image_size
    box_w = max(0.0, float(operator_box[2] - operator_box[0]))
    box_h = max(0.0, float(operator_box[3] - operator_box[1]))
    box_area = box_w * box_h
    frame_area = float(width * height)
    return (
        (box_area / frame_area) > OVERSIZED_BOX_AREA_RATIO
        or (box_w / width) > OVERSIZED_BOX_WIDTH_RATIO
        or (box_h / height) > OVERSIZED_BOX_HEIGHT_RATIO
    )


def generic_box_is_oversized(
    box: np.ndarray,
    image_size: Tuple[int, int],
    max_area_ratio: float,
    max_width_ratio: float,
    max_height_ratio: float,
) -> bool:
    width, height = image_size
    box_w = max(0.0, float(box[2] - box[0]))
    box_h = max(0.0, float(box[3] - box[1]))
    box_area = box_w * box_h
    frame_area = float(width * height)
    return (
        (box_area / frame_area) > max_area_ratio
        or (box_w / width) > max_width_ratio
        or (box_h / height) > max_height_ratio
    )


def face_has_visible_headroom(face_box: np.ndarray, operator_box: np.ndarray) -> bool:
    operator_h = max(1.0, float(operator_box[3] - operator_box[1]))
    headroom = float(face_box[1] - operator_box[1])
    return headroom >= operator_h * FACE_HEADROOM_RATIO


def face_is_strong_for_no_helmet(face_box: np.ndarray, operator_box: np.ndarray) -> bool:
    operator_w = max(1.0, float(operator_box[2] - operator_box[0]))
    operator_h = max(1.0, float(operator_box[3] - operator_box[1]))
    face_w = max(0.0, float(face_box[2] - face_box[0]))
    face_h = max(0.0, float(face_box[3] - face_box[1]))
    face_center_y = center_of(face_box)[1]
    relative_center_y = (face_center_y - float(operator_box[1])) / operator_h
    return (
        face_has_visible_headroom(face_box, operator_box)
        and relative_center_y <= STRONG_FACE_CENTER_Y_MAX_RATIO
        and (face_w / operator_w) >= STRONG_FACE_WIDTH_MIN_RATIO
        and (face_h / operator_h) >= STRONG_FACE_HEIGHT_MIN_RATIO
    )


def helmet_is_plausible_for_operator(helmet_box: np.ndarray, operator_box: np.ndarray) -> bool:
    head_box = head_zone_region(operator_box)
    return point_inside_box(center_of(helmet_box), head_box) or compute_iou(helmet_box, head_box) > 0.10


def load_label_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def infer_operator_state(
    operator_box: np.ndarray,
    operator_score: float,
    helmet_records: List[DetectionRecord],
    face_records: List[DetectionRecord],
    image_size: Tuple[int, int],
) -> OperatorDecision:
    upper_box = upper_body_region(operator_box)
    matched_faces = [
        face for face in face_records
        if point_inside_box(center_of(face.box), operator_box, margin_ratio=FACE_MARGIN_RATIO)
        and not generic_box_is_oversized(
            face.box,
            image_size,
            FACE_MAX_AREA_RATIO,
            FACE_MAX_WIDTH_RATIO,
            FACE_MAX_HEIGHT_RATIO,
        )
    ]
    matched_helmets = [
        helmet for helmet in helmet_records
        if (point_inside_box(center_of(helmet.box), upper_box) or compute_iou(helmet.box, upper_box) > 0.10)
        and helmet_is_plausible_for_operator(helmet.box, operator_box)
        and not generic_box_is_oversized(
            helmet.box,
            image_size,
            HELMET_MAX_AREA_RATIO,
            HELMET_MAX_WIDTH_RATIO,
            HELMET_MAX_HEIGHT_RATIO,
        )
    ]
    valid_faces = [face for face in matched_faces if face_has_visible_headroom(face.box, operator_box)]
    strong_faces = [face for face in valid_faces if face_is_strong_for_no_helmet(face.box, operator_box)]

    oversized_operator = operator_box_is_oversized(operator_box, image_size)
    diagnostics: List[str] = []
    if oversized_operator:
        diagnostics.append("FAIL: operator box too large")
    if not matched_faces:
        diagnostics.append("FAIL: no visible face in box")
    if matched_faces and not valid_faces:
        diagnostics.append("FAIL: face partial, top of head not visible")
    if valid_faces and not strong_faces:
        diagnostics.append("FAIL: face weak for high-confidence no-helmet")
    if strong_faces and not matched_helmets:
        diagnostics.append("FAIL: face seen, helmet missing")

    if matched_helmets:
        return OperatorDecision(
            raw_state="helmet",
            reason="helmet seen on upper body",
            operator_score=operator_score,
            face_records=valid_faces,
            helmet_records=matched_helmets,
            oversized_operator=oversized_operator,
            diagnostics=diagnostics,
        )
    if valid_faces:
        return OperatorDecision(
            raw_state="no_helmet",
            reason="face visible and helmet absent",
            operator_score=operator_score,
            face_records=valid_faces,
            helmet_records=matched_helmets,
            oversized_operator=oversized_operator,
            diagnostics=diagnostics,
        )
    return OperatorDecision(
        raw_state="unknown",
        reason="insufficient head visibility",
        operator_score=operator_score,
        face_records=valid_faces,
        helmet_records=matched_helmets,
        oversized_operator=oversized_operator,
        diagnostics=diagnostics,
    )


def draw_demo_frame(
    image: Image.Image,
    frame_idx: int,
    assignments: List[Tuple[int, DetectionRecord]],
    helmet_records: List[DetectionRecord],
    face_records: List[DetectionRecord],
    track_manager: TrackManager,
) -> Image.Image:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = load_label_font(LABEL_FONT_SIZE)
    image_w, image_h = annotated.size

    for helmet in helmet_records:
        draw.rectangle([tuple(helmet.box[:2]), tuple(helmet.box[2:])], outline="#33cc66", width=2)

    for face in face_records:
        draw.rectangle([tuple(face.box[:2]), tuple(face.box[2:])], outline="#00b3ff", width=2)

    for track_id, operator in assignments:
        decision = infer_operator_state(
            operator_box=operator.box,
            operator_score=operator.score,
            helmet_records=helmet_records,
            face_records=face_records,
            image_size=image.size,
        )
        state = track_manager.smoothed_state(track_id, decision.raw_state)

        color = {
            "helmet": "#33cc66",
            "no_helmet": "#ff4d4d",
            "unknown": "#ffb000",
        }[state]

        draw.rectangle([tuple(operator.box[:2]), tuple(operator.box[2:])], outline=color, width=3)
        label = {
            "helmet": "helmet",
            "no_helmet": "no helmet",
            "unknown": "unknown",
        }[state]
        box_x1 = float(operator.box[0])
        box_y1 = float(operator.box[1])
        box_y2 = float(operator.box[3])
        label_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = label_bbox[2] - label_bbox[0]
        text_h = label_bbox[3] - label_bbox[1]
        label_w = text_w + (2 * LABEL_PADDING_X)
        label_h = text_h + (2 * LABEL_PADDING_Y)

        x1 = min(max(0.0, box_x1), max(0.0, image_w - label_w))
        above_y = box_y1 - label_h - LABEL_MARGIN
        inside_y = box_y1 + LABEL_MARGIN
        below_y = box_y2 + LABEL_MARGIN

        if above_y >= 0:
            y1 = above_y
        elif inside_y + label_h <= image_h:
            y1 = inside_y
        else:
            y1 = max(0.0, min(below_y, image_h - label_h))

        draw.rectangle([x1, y1, x1 + label_w, y1 + label_h], fill=(0, 0, 0))
        draw.text((x1 + LABEL_PADDING_X, y1 + LABEL_PADDING_Y), label, fill=color, font=font)

    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 helmet demo inference on a video.")
    parser.add_argument("--video-path", default=VIDEO_PATH, help="Input video path.")
    parser.add_argument("--output-path", default=OUTPUT_PATH, help="Output demo video path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = None
    temp_mp4_path = None
    processing_video_path = video_path

    if video_path.suffix.lower() == ".mkv":
        temp_dir = Path(tempfile.mkdtemp())
        temp_mp4_path = convert_mkv_to_mp4(video_path, temp_dir)
        processing_video_path = temp_mp4_path

    try:
        accelerator = Accelerator()
        device = accelerator.device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        print(f"Torch device: {device}")
        print("Loading SAM3 Video model...")
        model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
        processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
        print("Model loaded.")

        video_frames, _ = load_video(str(processing_video_path))
        print(f"Loaded {len(video_frames)} frames.")

        if FRAME_SKIP > 1:
            video_frames = video_frames[::FRAME_SKIP]
            print(f"Frame skip applied. Using {len(video_frames)} frames.")

        if MAX_RESOLUTION is not None:
            video_frames = resize_frames(video_frames, MAX_RESOLUTION)
            print(f"Frames resized to max resolution {MAX_RESOLUTION}.")

        outputs = process_video_in_chunks(
            video_frames=video_frames,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
        )

        track_manager = TrackManager()
        frames_to_write: List[np.ndarray] = []

        for idx, frame in enumerate(video_frames):
            frame_pil = frame.convert("RGB") if isinstance(frame, Image.Image) else Image.fromarray(frame).convert("RGB")
            operator_records = frame_outputs_to_records(outputs["operator"].get(idx), OPERATOR_CONF)
            helmet_records = frame_outputs_to_records(outputs["helmet"].get(idx), HELMET_CONF)
            face_records = frame_outputs_to_records(outputs["face"].get(idx), FACE_CONF)
            operator_records = [
                record for record in operator_records
                if not operator_box_is_oversized(record.box, frame_pil.size)
            ]

            assignments = track_manager.update(operator_records)
            annotated = draw_demo_frame(
                image=frame_pil,
                frame_idx=idx,
                assignments=assignments,
                helmet_records=helmet_records,
                face_records=face_records,
                track_manager=track_manager,
            )
            frames_to_write.append(np.array(annotated))

        print(f"Writing demo video to: {output_path}")
        iio.imwrite(output_path, frames_to_write, fps=OUTPUT_FPS, codec="libx264")
        print("Done.")
    finally:
        if temp_mp4_path and temp_mp4_path.exists():
            temp_mp4_path.unlink()
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
