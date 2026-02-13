import os, re, json
from pathlib import Path
from typing import List, Optional, Tuple

import av
import cv2
import numpy as np
import torch
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor

# ==============================================================================
# CONFIG
# ==============================================================================
ROOT_DIR = Path("/home/evox5090ia/Downloads/2026-01-12_09-46-12/video")  # <-- ALTERAR
# SUBFOLDERS = ["direita", "esquerda"]
SUBFOLDERS = ["esquerda"]

PROMPT = "waste container"
CONF_THRES = 0.75

# Performance knobs (ajuste conforme necessário)
FRAME_STRIDE = 3          # usa 1 em cada N frames (3 => 0,3,6,9,...)
MAX_FRAMES = 180          # máximo de frames analisados por vídeo (após stride)
MAX_VIDEO_SECONDS = 180   # opcional: ignora vídeos muito longos

OUT_TXT = ROOT_DIR / "sam3_hits.txt"

# ==============================================================================
# LOAD SAM3
# ==============================================================================
accelerator = Accelerator()
device = accelerator.device
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

print(f"Loading SAM3 Video model on device={device}, dtype={dtype} ...")
sam3_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
sam3_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
print("SAM3 loaded.\n")

# ==============================================================================
# HELPERS
# ==============================================================================
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

def decode_video_sampled_frames(video_path: Path, stride: int, max_frames: int) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        idx = -1
        for packet in container.demux(stream):
            for frame in packet.decode():
                idx += 1
                if stride > 1 and (idx % stride) != 0:
                    continue
                img = frame.to_ndarray(format="bgr24")
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                frames.append(img)
                if len(frames) >= max_frames:
                    return frames
    return frames

def video_has_prompt_detection(
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    frames_bgr: List[np.ndarray],
    prompt: str,
    conf_thresh: float,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    if not frames_bgr:
        return False

    inference_session = processor.init_video_session(
        video=frames_bgr,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    processor.add_text_prompt(inference_session=inference_session, text=[prompt])

    num_frames = len(frames_bgr)

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=num_frames,
    ):
        processed = processor.postprocess_outputs(inference_session, model_outputs)

        boxes = processed.get("boxes", None)
        scores = processed.get("scores", None)
        object_ids = processed.get("object_ids", None)
        prompt_to_obj_ids = processed.get("prompt_to_obj_ids", {}) or {}

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

        # Early stop: basta 1 bbox do prompt acima do threshold
        for score_t, oid_t in zip(scores_cpu, object_ids_cpu):
            score = float(score_t.item())
            oid = int(oid_t.item())
            if oid in obj_ids_for_prompt and score >= conf_thresh:
                return True

    return False

# ==============================================================================
# MAIN
# ==============================================================================
hits: List[str] = []
total = 0

for sub in SUBFOLDERS:
    subdir = ROOT_DIR / sub
    if not subdir.exists():
        print(f"Warning: subfolder not found: {subdir}")
        continue

    video_files = sorted(subdir.glob("*.mkv"))
    for video_path in video_files:
        total += 1
        print(f"[{sub}] Checking: {video_path.name}")

        dur = get_video_duration_seconds(video_path)
        if dur > 0 and dur > MAX_VIDEO_SECONDS:
            print(f"  - Skipping (duration {dur:.1f}s > {MAX_VIDEO_SECONDS}s)")
            continue

        frames = decode_video_sampled_frames(video_path, stride=FRAME_STRIDE, max_frames=MAX_FRAMES)
        if not frames:
            print("  - No frames decoded, skipping.")
            continue

        has_det = video_has_prompt_detection(
            model=sam3_model,
            processor=sam3_processor,
            frames_bgr=frames,
            prompt=PROMPT,
            conf_thresh=CONF_THRES,
            device=device,
            dtype=dtype,
        )

        if has_det:
            line = f"{sub}_{video_path.name}"
            hits.append(line)
            print(f"  + HIT: {line}")
        else:
            print("  - no detections")

# Write output
hits = sorted(set(hits))
with open(OUT_TXT, "w", encoding="utf-8") as f:
    for line in hits:
        f.write(line + "\n")

print("\nDone.")
print(f"Videos checked: {total}")
print(f"Hits: {len(hits)}")
print(f"Saved: {OUT_TXT}")
