import av, cv2, numpy as np, os, json, torch, re
from pathlib import Path
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor

# ==============================================================================
# --- ESSENTIAL HYPERPARAMETER CONFIGURATION ---
# ==============================================================================

INPUT_DIR = Path("/home/evox5090ia/sumasaojoao/2025-12-22_08-45-47/video/esquerda/")
OUTPUT_DIR = Path("/home/evox5090ia/sumasaojoao/2025-12-22_08-45-47/video/esquerda/")

# NEW: Start processing only from this source video id (inclusive).
# Example: 1063 means it will process 1063, 1064, 1065, ...
START_FROM_SOURCE_ID = 0  # set this as needed

# SAM3 prompt (as requested)
SAM3_PROMPT = "waste container"

# Detection / processing thresholds
CONF_THRES = 0.40  # applies to SAM3 scores
BUFFER_BEFORE = 20
BUFFER_AFTER = 20
EXPORT_CRF = 20
EXPORT_CODEC = "libx264"
SCENE_THR = 0.985

# Skip long videos
MAX_VIDEO_SECONDS = 120  # seconds

# NEW: Also export a visualization video with bboxes
EXPORT_SHOWDET = True
SHOWDET_FILENAME = "video_showdet.mp4"  # created alongside video.mkv
SHOWDET_CODEC = "libx264"
SHOWDET_CRF = 28
SHOWDET_PRESET = "ultrafast"
SHOWDET_SCALE_WIDTH = None  # e.g. 960 to downscale for lighter CPU, or None to keep original

# Internal
EFFECTIVE_GAP = BUFFER_BEFORE + BUFFER_AFTER - 1
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# --- LOAD SAM3 (pretrained) ---
# ==============================================================================

accelerator = Accelerator()
device = accelerator.device
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

print(f"Loading SAM3 Video model on device={device}, dtype={dtype} ...")
sam3_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
sam3_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
print("SAM3 loaded.\n")


# ==============================================================================
# --- UTILITY FUNCTIONS ---
# ==============================================================================

def is_same_scene(a: np.ndarray, b: np.ndarray, thr: float = SCENE_THR) -> bool:
    g1, g2 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([g1], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([g2], [0], None, [64], [0, 256])
    h1 = cv2.normalize(h1, h1).flatten()
    h2 = cv2.normalize(h2, h2).flatten()
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) > thr


def extract_source_video_id(video_path: Path) -> Optional[int]:
    m = re.search(r"_([0-9]+)\.mkv$", video_path.name)
    return int(m.group(1)) if m else None


def get_video_duration_seconds(video_path: Path) -> float:
    try:
        with av.open(str(video_path)) as c:
            s = c.streams.video[0]
            if c.duration is not None:
                return float(c.duration) / 1_000_000.0
            if s.duration is not None and s.time_base is not None:
                return float(s.duration * s.time_base)
            if s.frames and s.average_rate:
                return float(s.frames / float(s.average_rate))
    except Exception:
        pass
    return -1.0


def estimate_duration_by_partial_decode(video_path: Path, max_frames: int = 300) -> float:
    try:
        with av.open(str(video_path)) as c:
            s = c.streams.video[0]
            time_base = float(s.time_base) if s.time_base is not None else None
            if time_base is None:
                return -1.0

            pts_first, pts_last = None, None
            decoded = 0
            for pkt in c.demux(s):
                for frame in pkt.decode():
                    if frame.pts is None:
                        continue
                    if pts_first is None:
                        pts_first = frame.pts
                    pts_last = frame.pts
                    decoded += 1
                    if decoded >= max_frames:
                        break
                if decoded >= max_frames:
                    break

            if pts_first is None or pts_last is None or pts_last <= pts_first:
                return -1.0

            sampled_seconds = float(pts_last - pts_first) * time_base
            if decoded < max_frames:
                return sampled_seconds

            if s.average_rate:
                fps = float(s.average_rate)
                return float(s.frames / fps) if s.frames else -1.0

            return -1.0
    except Exception:
        return -1.0


# -----------------------------
# MINIMAL FIX: safe FPS rate
# -----------------------------
def _safe_rate_from_stream(v, default: Fraction = Fraction(25, 1)) -> Fraction:
    r = getattr(v, "average_rate", None) or getattr(v, "base_rate", None)
    if r is None:
        return default
    if hasattr(r, "numerator") and hasattr(r, "denominator"):
        num = int(r.numerator)
        den = int(r.denominator)
        if den != 0 and num > 0:
            return Fraction(num, den)
        return default
    try:
        return Fraction(str(float(r)))
    except Exception:
        return default


# -----------------------------
# NEW: resume output folder id
# -----------------------------
def get_next_video_id(output_dir: Path, required_file: str = "video.mkv") -> int:
    """
    Returns next clip folder id to use (max existing numeric folder + 1).
    Only counts numeric folders that contain required_file (default: video.mkv).
    """
    max_id = 0
    if not output_dir.exists():
        return 1

    for p in output_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.isdigit():
            continue
        if required_file and not (p / required_file).exists():
            continue
        max_id = max(max_id, int(p.name))

    return max_id + 1


def export_segment(video_path: Path, keep_pts: np.ndarray, out_path: Path,
                   codec: str = EXPORT_CODEC, crf: int = EXPORT_CRF):
    keep = set(int(x) for x in keep_pts)
    with av.open(str(video_path)) as ic:
        v = ic.streams.video[0]
        rate = _safe_rate_from_stream(v)

        oc = av.open(str(out_path), "w")
        ov = oc.add_stream(codec, rate=rate)
        ov.width, ov.height, ov.pix_fmt = v.width, v.height, "yuv420p"
        ov.options = {"preset": "6", "crf": str(crf)}
        for pkt in ic.demux(v):
            for f in pkt.decode():
                if f.pts in keep:
                    img = f.to_ndarray(format="bgr24")
                    frm = av.VideoFrame.from_ndarray(img, format="bgr24")
                    for p in ov.encode(frm):
                        oc.mux(p)
        for p in ov.encode(None):
            oc.mux(p)
        oc.close()


def maybe_resize(img: np.ndarray, target_w: Optional[int]) -> np.ndarray:
    if target_w is None:
        return img
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    scale = target_w / float(w)
    new_h = int(h * scale)
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def draw_detections(img: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        conf = float(d.get("conf", 0.0))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{d.get('class','obj')} {conf:.2f}",
                    (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def export_showdet_video_from_source(
    source_video_path: Path,
    keep_pts: np.ndarray,
    seg_idx: List[int],
    frame_detections: Dict[int, List[Dict[str, Any]]],
    out_path: Path,
):
    keep = set(int(x) for x in keep_pts)
    pts_to_origidx = {int(p): int(idx) for p, idx in zip(keep_pts.tolist(), seg_idx)}

    with av.open(str(source_video_path)) as ic:
        v = ic.streams.video[0]
        src_rate = _safe_rate_from_stream(v)

        oc = av.open(str(out_path), "w")
        ov = oc.add_stream(SHOWDET_CODEC, rate=src_rate)
        ov.options = {"preset": SHOWDET_PRESET, "crf": str(SHOWDET_CRF)}

        initialized = False

        for pkt in ic.demux(v):
            for f in pkt.decode():
                if f.pts not in keep:
                    continue

                img = f.to_ndarray(format="bgr24")

                orig_idx = pts_to_origidx.get(int(f.pts), None)
                if orig_idx is not None and orig_idx in frame_detections:
                    img = draw_detections(img, frame_detections[orig_idx])

                img = maybe_resize(img, SHOWDET_SCALE_WIDTH)

                if not initialized:
                    ov.width, ov.height = img.shape[1], img.shape[0]
                    ov.pix_fmt = "yuv420p"
                    initialized = True

                frm = av.VideoFrame.from_ndarray(img, format="bgr24")
                for p in ov.encode(frm):
                    oc.mux(p)

        for p in ov.encode(None):
            oc.mux(p)

        oc.close()


def decode_video_to_frames_and_pts(video_path: Path) -> Tuple[List[np.ndarray], List[int]]:
    frames: List[np.ndarray] = []
    pts_all: List[int] = []

    container = av.open(str(video_path))
    stream = container.streams.video[0]

    frame_idx = -1
    for packet in container.demux(stream):
        for frame in packet.decode():
            frame_idx += 1
            pts_all.append(int(frame.pts if frame.pts is not None else frame_idx))
            img = frame.to_ndarray(format="bgr24")
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            frames.append(img)

    container.close()
    return frames, pts_all


def run_sam3_for_prompt(
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    device: torch.device,
    dtype: torch.dtype,
    frames_bgr: List[np.ndarray],
    prompt: str,
    conf_thresh: float,
) -> Tuple[List[int], Dict[int, List[Dict[str, Any]]]]:
    if not frames_bgr:
        return [], {}

    inference_session = processor.init_video_session(
        video=frames_bgr,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    processor.add_text_prompt(inference_session=inference_session, text=[prompt])

    detections_frame_idxs: List[int] = []
    frame_detections: Dict[int, List[Dict[str, Any]]] = {}

    num_frames = len(frames_bgr)
    print(f"  Running SAM3 propagation over {num_frames} frames with prompt='{prompt}' ...")

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=num_frames,
    ):
        frame_idx = int(model_outputs.frame_idx)
        processed = processor.postprocess_outputs(inference_session, model_outputs)

        boxes = processed.get("boxes", None)
        scores = processed.get("scores", None)
        object_ids = processed.get("object_ids", None)
        prompt_to_obj_ids = processed.get("prompt_to_obj_ids", {}) or {}

        frame_detections[frame_idx] = []

        if boxes is None or scores is None or object_ids is None:
            continue
        if getattr(boxes, "numel", lambda: 0)() == 0:
            continue

        obj_ids_for_prompt = prompt_to_obj_ids.get(prompt, [])
        if isinstance(obj_ids_for_prompt, torch.Tensor):
            obj_ids_for_prompt = obj_ids_for_prompt.detach().cpu().tolist()
        obj_ids_for_prompt = set(int(x) for x in obj_ids_for_prompt)

        if not obj_ids_for_prompt:
            continue

        scores_cpu = scores.detach().cpu()
        object_ids_cpu = object_ids.detach().cpu()
        boxes_cpu = boxes.detach().cpu()

        for box_t, score_t, oid_t in zip(boxes_cpu, scores_cpu, object_ids_cpu):
            score = float(score_t.item())
            oid = int(oid_t.item())
            if score < conf_thresh:
                continue
            if oid not in obj_ids_for_prompt:
                continue

            x1, y1, x2, y2 = [float(v) for v in box_t.tolist()]
            frame_detections[frame_idx].append({
                "class": prompt,
                "bbox": [x1, y1, x2, y2],
                "conf": score,
                "object_id": oid,
            })

        if frame_detections[frame_idx]:
            detections_frame_idxs.append(frame_idx)

    frame_detections = {k: v for k, v in frame_detections.items() if v}
    print(f"  SAM3: {len(detections_frame_idxs)} frames with detections for prompt='{prompt}'.")
    return detections_frame_idxs, frame_detections


# ==============================================================================
# --- MAIN LOOP ---
# ==============================================================================

video_files = list(INPUT_DIR.glob("*.mkv"))

def sort_key(p: Path):
    sid = extract_source_video_id(p)
    return (sid is None, sid if sid is not None else p.name)

video_files = sorted(video_files, key=sort_key)

# NEW: resume from last generated output folder (instead of resetting to 1)
video_id = get_next_video_id(OUTPUT_DIR, required_file="video.mkv")
print(f"Resuming: next output video_id folder will be {video_id}")

for video_path in video_files:
    source_id = extract_source_video_id(video_path)
    if source_id is None:
        print(f"\nSkipping {video_path.name}: could not extract trailing numeric id for ordering.")
        continue

    # NEW: skip until we reach START_FROM_SOURCE_ID
    if source_id < START_FROM_SOURCE_ID:
        continue

    print(f"\nProcessing {video_path.name} (source_id={source_id}, SAM3 prompt: '{SAM3_PROMPT}')")

    duration_s = get_video_duration_seconds(video_path)
    if duration_s < 0:
        duration_s = estimate_duration_by_partial_decode(video_path, max_frames=300)

    if duration_s > 0 and duration_s > MAX_VIDEO_SECONDS:
        print(f"Skipping {video_path.name}: duration {duration_s:.2f}s > {MAX_VIDEO_SECONDS}s")
        continue

    frames_bgr, frame_pts_all = decode_video_to_frames_and_pts(video_path)
    total_frames = len(frames_bgr)

    if total_frames == 0:
        print("Empty video, skipping.")
        continue

    detections, frame_detections = run_sam3_for_prompt(
        model=sam3_model,
        processor=sam3_processor,
        device=device,
        dtype=dtype,
        frames_bgr=frames_bgr,
        prompt=SAM3_PROMPT,
        conf_thresh=CONF_THRES,
    )

    frames_bgr = []

    if not detections:
        print("No relevant detections found, skipping video.")
        continue

    detections = sorted(set(detections))
    segs: List[Tuple[int, int]] = []
    s, p = detections[0], detections[0]
    for f in detections[1:]:
        if f - p > EFFECTIVE_GAP:
            segs.append((s, p))
            s = f
        p = f
    segs.append((s, p))

    last_cut_last_frame = None
    for s, e in segs:
        s_buf, e_buf = max(0, s - BUFFER_BEFORE), min(total_frames - 1, e + BUFFER_AFTER)
        seg_idx = sorted(set(range(s_buf, e_buf + 1)))
        keep_pts = np.array([frame_pts_all[idx] for idx in seg_idx], dtype=np.int64)
        if len(keep_pts) == 0:
            continue

        with av.open(str(video_path)) as ic_check:
            v_check = ic_check.streams.video[0]
            first, last = None, None
            for pkt in ic_check.demux(v_check):
                for f in pkt.decode():
                    if f.pts == int(keep_pts[0]):
                        first = f.to_ndarray(format="bgr24")
                    if f.pts == int(keep_pts[-1]):
                        last = f.to_ndarray(format="bgr24")
                    if first is not None and last is not None:
                        break
                if first is not None and last is not None:
                    break

        if last_cut_last_frame is not None and first is not None:
            if is_same_scene(last_cut_last_frame, first):
                print(f"Skipping static duplicate (videoid {video_id})")
                continue

        out_dir = OUTPUT_DIR / f"{video_id}"
        os.makedirs(out_dir, exist_ok=True)
        out_vid = out_dir / "video.mkv"
        out_pts = out_dir / "frame_pts.npy"
        out_json = out_dir / "frame_detections.json"
        out_showdet = out_dir / SHOWDET_FILENAME

        np.save(out_pts, keep_pts)
        export_segment(video_path, keep_pts, out_vid)

        clip_detections = {str(idx): frame_detections[idx] for idx in seg_idx if idx in frame_detections}
        with open(out_json, "w") as f:
            json.dump(clip_detections, f, indent=4)

        if EXPORT_SHOWDET:
            export_showdet_video_from_source(
                source_video_path=video_path,
                keep_pts=keep_pts,
                seg_idx=seg_idx,
                frame_detections=frame_detections,
                out_path=out_showdet,
            )

        last_cut_last_frame = last
        print(f"Saved videoid {video_id} (source_id={source_id}) ({len(keep_pts)} frames, {len(clip_detections)} frames with detections)")
        video_id += 1

print("\nAll videos processed ✅")
