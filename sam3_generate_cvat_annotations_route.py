"""
SCRIPT OVERVIEW (how to use + what it does)

Goal
- For each video folder (video/<camera>/<videoid>/), run SAM3 Video with text prompts, export CVAT XML boxes,
  and create an annotated MP4 preview.

Dataset layout expected
- DATASET_ROOT/
    video/
      direita/
        <videoid>/
          video.mp4  (preferred) OR video.mkv
      esquerda/
        <videoid>/
          video.mp4  (preferred) OR video.mkv

Processing rules
- SKIP rule: if <videoid>/annotations.xml already exists, that folder is not processed.
- Input selection:
  - If video.mp4 exists: use it.
  - Else if video.mkv exists: remux to _sam3_tmp_video.mp4 (ffmpeg -c copy), use that, then delete it.
- Outputs per <videoid>/ folder:
  - annotations.xml  (CVAT 1.1, tracks with per-frame boxes)
  - annotated.mp4    (original-resolution frames with boxes + labels overlay)

SAM3 prompts / labels
- PROMPTS maps:
  - "person" -> "pessoa"
  - "license plates" -> "matricula"
  - "waste container" -> "waste container"
- CONF_THRESH filters boxes per-frame by score.

GPU memory strategy (to avoid stopping on OOM)
- For short clips (< DOWNSCALE_LONG_FRAMES_THRESHOLD frames): run full resolution only (no resizing).
- For long clips (>= threshold): try full resolution first; if CUDA OutOfMemoryError occurs, retry with
  progressively smaller max-side targets: 1280, 1024, 896, 768, 640.
- When downscaling is used:
  - Frames are resized for inference.
  - Boxes are scaled back to original coordinates using sx = orig_w / resized_w and sy = orig_h / resized_h,
    so annotations.xml always matches the original video resolution.
- After OOM and after each folder: torch.cuda.empty_cache() is called to reduce carry-over pressure.

Notes
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set by default (helps fragmentation on some setups).
"""




#!/usr/bin/env python
from pathlib import Path
from typing import Dict, Any, List, Tuple
import subprocess
import os

import numpy as np
import torch
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
import xml.etree.ElementTree as ET
import supervision as sv
from PIL import Image
import imageio.v3 as iio


# ================== HARD-CODED CONFIG ==================

DATASET_ROOT = Path("/home/evox5090ia/rotasinprogress_sumasaojoao/2026-03-18_13-56-31").resolve()

VIDEO_MKV_NAME = "video.mkv"
VIDEO_MP4_NAME = "video.mp4"
TEMP_MP4_NAME = "_sam3_tmp_video.mp4"

OUTPUT_XML_NAME = "annotations_sam.xml"
OUTPUT_VIDEO_NAME = "annotated_sam.mp4"

CONF_THRESH = 0.05

PROMPTS = [
    ("person", "pessoa", 0),
    ("license plates", "matricula", 1),
    ("waste container", "waste container", 20),
]

COLOR = sv.ColorPalette.from_hex(
    [
        "#ffff00", "#ff9b00", "#ff8080", "#ff66b2",
        "#ff66ff", "#b266ff", "#9999ff", "#3399ff",
        "#66ffff", "#33ff99", "#66ff66", "#99ff00",
    ]
)

# Only attempt downscale retries when clip is "long"
DOWNSCALE_LONG_FRAMES_THRESHOLD = 180

# Progressive fallbacks (max(w,h) target). We will retry smaller on OOM.
DOWNSCALE_MAX_SIDES_LONG = [1280, 1024, 896, 768, 640]

# Optional: helps with fragmentation on some setups (safe no-op if already set)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# =======================================================


def convert_mkv_to_temp_mp4(mkv_path: Path, temp_mp4_path: Path) -> None:
    if temp_mp4_path.exists():
        temp_mp4_path.unlink()
    cmd = ["ffmpeg", "-y", "-i", str(mkv_path), "-c", "copy", str(temp_mp4_path)]
    print(f"  Converting MKV to MP4: {mkv_path.name} -> {temp_mp4_path.name}")
    subprocess.run(cmd, check=True)
    print("  Conversion done.")


def select_video_path(video_dir: Path) -> Tuple[Path | None, bool]:
    mp4_path = video_dir / VIDEO_MP4_NAME
    mkv_path = video_dir / VIDEO_MKV_NAME
    temp_mp4_path = video_dir / TEMP_MP4_NAME

    if mp4_path.exists():
        return mp4_path, False
    if mkv_path.exists():
        convert_mkv_to_temp_mp4(mkv_path, temp_mp4_path)
        return temp_mp4_path, True
    return None, False


def _to_uint8_np_frames(video_frames: List[Any]) -> List[np.ndarray]:
    np_frames: List[np.ndarray] = []
    for f in video_frames:
        if isinstance(f, np.ndarray):
            if f.dtype != np.uint8:
                f = f.astype(np.uint8)
            np_frames.append(f)
        else:
            np_frames.append(np.array(f, dtype=np.uint8))
    return np_frames


def resize_frames_to_max_side(frames: List[np.ndarray], max_side: int) -> Tuple[List[np.ndarray], float, float]:
    """
    Resize frames so that max(h,w) == max_side (if larger). Returns resized_frames and scale factors:
      sx = original_w / resized_w
      sy = original_h / resized_h
    """
    if not frames:
        return frames, 1.0, 1.0

    h, w = frames[0].shape[:2]
    if max(h, w) <= max_side:
        return frames, 1.0, 1.0

    scale = max_side / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    try:
        import cv2  # type: ignore
        resized = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
    except Exception:
        resized = [np.array(Image.fromarray(f).resize((new_w, new_h), resample=Image.BILINEAR)) for f in frames]

    sx = w / float(new_w)
    sy = h / float(new_h)
    print(f"  [INFO] Downscaling for SAM3: {w}x{h} -> {new_w}x{new_h} (sx={sx:.4f}, sy={sy:.4f})")
    return resized, sx, sy


def run_sam3_on_video(
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    device: torch.device,
    dtype: torch.dtype,
    video_path: Path,
    conf_thresh: float,
) -> Tuple[Dict[int, Dict[str, Any]], List[np.ndarray], float]:
    video_frames, meta = load_video(str(video_path))
    num_frames = len(video_frames)
    print(f"  Loaded {num_frames} frames from {video_path.name}")
    fps = float(meta.get("fps", 10.0))

    # Original frames (for annotated.mp4); boxes will be rescaled back to this coordinate space.
    original_frames = _to_uint8_np_frames(video_frames)

    # Decide retry strategy:
    # - For short clips, try full-res only (avoid unnecessary rescale + quality loss)
    # - For long clips, try progressively smaller max-side values if we hit OOM
    if num_frames < DOWNSCALE_LONG_FRAMES_THRESHOLD:
        candidate_max_sides: List[int | None] = [None]  # None => no resize
    else:
        candidate_max_sides = [None] + DOWNSCALE_MAX_SIDES_LONG

    last_oom: Exception | None = None

    for max_side in candidate_max_sides:
        # Prepare frames for SAM3 this attempt
        if max_side is None:
            np_frames = original_frames
            sx = sy = 1.0
            if num_frames >= DOWNSCALE_LONG_FRAMES_THRESHOLD:
                print("  [INFO] Attempting SAM3 at full resolution first (long clip).")
        else:
            np_frames, sx, sy = resize_frames_to_max_side(original_frames, int(max_side))

        # Build session and run propagation; on OOM, cleanup and retry smaller.
        try:
            inference_session = processor.init_video_session(
                video=np_frames,
                inference_device=device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=dtype,
            )

            prompt_texts = [p[0] for p in PROMPTS]
            prompt_to_xml_label = {p[0]: p[1] for p in PROMPTS}
            processor.add_text_prompt(inference_session=inference_session, text=prompt_texts)

            outputs_per_frame: Dict[int, Dict[str, Any]] = {}
            print("  Running SAM3 propagation...")

            for model_outputs in model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=num_frames,
            ):
                processed = processor.postprocess_outputs(inference_session, model_outputs)

                boxes = processed.get("boxes")
                scores = processed.get("scores")
                object_ids = processed.get("object_ids")
                prompt_to_obj_ids = processed.get("prompt_to_obj_ids", {})

                if torch.is_tensor(boxes):
                    boxes = boxes.detach().cpu()
                if torch.is_tensor(scores):
                    scores = scores.detach().cpu()
                if torch.is_tensor(object_ids):
                    object_ids = object_ids.detach().cpu()

                p2o_cpu: Dict[str, List[int]] = {}
                for k, v in (prompt_to_obj_ids or {}).items():
                    if torch.is_tensor(v):
                        p2o_cpu[k] = v.detach().cpu().tolist()
                    else:
                        p2o_cpu[k] = [int(x) for x in v]

                outputs_per_frame[int(model_outputs.frame_idx)] = {
                    "boxes": boxes,
                    "scores": scores,
                    "object_ids": object_ids,
                    "prompt_to_obj_ids": p2o_cpu,
                }

                del processed

            print(f"  SAM3 processed {len(outputs_per_frame)} frames")

            # Build tracks, rescaling boxes back to original coordinates via sx, sy
            tracks: Dict[int, Dict[str, Any]] = {}
            object_id_to_label: Dict[int, str] = {}

            for frame_idx in sorted(outputs_per_frame.keys()):
                fo = outputs_per_frame[frame_idx]
                boxes = fo["boxes"]
                scores = fo["scores"]
                object_ids = fo["object_ids"]
                prompt_to_obj_ids = fo.get("prompt_to_obj_ids", {})

                if boxes is None or scores is None or object_ids is None:
                    continue
                if hasattr(boxes, "numel") and boxes.numel() == 0:
                    continue

                frame_objid_to_label: Dict[int, str] = {}
                for prompt, obj_ids_for_prompt in (prompt_to_obj_ids or {}).items():
                    xml_label = prompt_to_xml_label.get(prompt)
                    if xml_label is None:
                        continue
                    for oid in obj_ids_for_prompt:
                        frame_objid_to_label[int(oid)] = xml_label

                keep_mask = scores > conf_thresh
                if torch.is_tensor(keep_mask) and int(keep_mask.sum().item()) == 0:
                    continue

                sel_boxes = boxes[keep_mask]
                sel_scores = scores[keep_mask]
                sel_obj_ids = object_ids[keep_mask]

                for oid_t, box_t, score_t in zip(sel_obj_ids, sel_boxes, sel_scores):
                    oid = int(oid_t.item()) if hasattr(oid_t, "item") else int(oid_t)

                    label = object_id_to_label.get(oid)
                    if label is None:
                        label = frame_objid_to_label.get(oid)
                        if label is None:
                            continue
                        object_id_to_label[oid] = label

                    xtl, ytl, xbr, ybr = [float(v) for v in box_t.tolist()]
                    score = float(score_t.item()) if hasattr(score_t, "item") else float(score_t)

                    # Rescale back to original
                    xtl *= sx
                    xbr *= sx
                    ytl *= sy
                    ybr *= sy

                    if oid not in tracks:
                        tracks[oid] = {"label": label, "boxes": []}
                    tracks[oid]["boxes"].append(
                        {
                            "frame": int(frame_idx),
                            "xtl": xtl,
                            "ytl": ytl,
                            "xbr": xbr,
                            "ybr": ybr,
                            "score": score,
                        }
                    )

            print(f"  Collected {len(tracks)} tracks (pessoa/matricula/waste container)")
            return tracks, original_frames, fps

        except torch.OutOfMemoryError as e:
            last_oom = e
            print(f"  [OOM] CUDA out of memory at max_side={max_side}. Retrying with smaller resolution...")
            # Cleanup between attempts
            try:
                del inference_session
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    # If we got here, all attempts failed
    raise last_oom if last_oom is not None else RuntimeError("SAM3 failed for unknown reasons.")


def build_cvat_xml(tracks: Dict[int, Dict[str, Any]]) -> ET.ElementTree:
    root = ET.Element("annotations")
    version_el = ET.SubElement(root, "version")
    version_el.text = "1.1"

    meta_el = ET.SubElement(root, "meta")
    task_el = ET.SubElement(meta_el, "task")
    mode_el = ET.SubElement(task_el, "mode")
    mode_el.text = "annotation"
    labels_el = ET.SubElement(task_el, "labels")

    label_names = sorted({t["label"] for t in tracks.values()}) if tracks else []
    for name in label_names:
        label_el = ET.SubElement(labels_el, "label")
        name_el = ET.SubElement(label_el, "name")
        name_el.text = name

    for oid in sorted(tracks.keys()):
        track_data = tracks[oid]
        track_el = ET.SubElement(root, "track", id=str(oid), label=track_data["label"], source="file")
        boxes_sorted: List[Dict[str, Any]] = sorted(track_data["boxes"], key=lambda b: b["frame"])
        for box in boxes_sorted:
            attrs = {
                "frame": str(box["frame"]),
                "keyframe": "1",
                "outside": "0",
                "occluded": "0",
                "xtl": f"{box['xtl']:.6f}",
                "ytl": f"{box['ytl']:.6f}",
                "xbr": f"{box['xbr']:.6f}",
                "ybr": f"{box['ybr']:.6f}",
                "z_order": "0",
            }
            ET.SubElement(track_el, "box", **attrs)

    return ET.ElementTree(root)


def annotate_frame_with_supervision(
    frame: np.ndarray,
    detections: sv.Detections,
    trackid_to_label: Dict[int, str],
) -> np.ndarray:
    if len(detections) == 0:
        return frame

    pil_img = Image.fromarray(frame)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=pil_img.size)

    box_annotator = sv.BoxAnnotator(color=COLOR, color_lookup=sv.ColorLookup.INDEX, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=text_scale,
        text_padding=5,
        text_color=sv.Color.BLACK,
        text_thickness=1,
    )

    labels: List[str] = []
    for track_id, score in zip(detections.class_id, detections.confidence):
        tid = int(track_id)
        cls_name = trackid_to_label.get(tid, "unknown")
        labels.append(f"id={tid} | {cls_name} {score:.2f}" if score is not None else f"id={tid} | {cls_name}")

    annotated = frame.copy()
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    return annotated


def create_annotated_video(
    tracks: Dict[int, Dict[str, Any]],
    video_frames: List[np.ndarray],
    out_video_path: Path,
    fps: float,
):
    print(f"  Creating annotated video: {out_video_path.name}")

    trackid_to_label = {tid: tdata["label"] for tid, tdata in tracks.items()}

    frame_to_dets: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for tid, tdata in tracks.items():
        for box in tdata["boxes"]:
            fidx = box["frame"]
            frame_to_dets.setdefault(fidx, []).append((tid, box))

    annotated_frames: List[np.ndarray] = []
    for frame_idx, frame in enumerate(video_frames):
        dets_this_frame = frame_to_dets.get(frame_idx, [])
        if not dets_this_frame:
            annotated_frames.append(frame)
            continue

        xyxy_list: List[List[float]] = []
        conf_list: List[float] = []
        class_id_list: List[int] = []

        for tid, box in dets_this_frame:
            xyxy_list.append([box["xtl"], box["ytl"], box["xbr"], box["ybr"]])
            conf_list.append(box["score"])
            class_id_list.append(tid)

        detections = sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            confidence=np.array(conf_list, dtype=np.float32),
            class_id=np.array(class_id_list, dtype=np.int32),
        )

        annotated_frames.append(annotate_frame_with_supervision(frame, detections, trackid_to_label))

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_video_path, annotated_frames, fps=fps, codec="libx264")
    print(f"  Annotated video saved: {out_video_path}")


def process_video_dir(
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    device: torch.device,
    dtype: torch.dtype,
    video_dir: Path,
):
    xml_path = video_dir / OUTPUT_XML_NAME
    out_video_path = video_dir / OUTPUT_VIDEO_NAME

    # Requirement: if annotations.xml exists, do not process
    if xml_path.exists():
        print(f"[SKIP] {video_dir} already has {OUTPUT_XML_NAME}")
        return

    video_path, is_temp = select_video_path(video_dir)
    if video_path is None:
        print(f"[WARN] No {VIDEO_MKV_NAME} or {VIDEO_MP4_NAME} in {video_dir}, skipping.")
        return

    try:
        print(f"[INFO] Processing video: {video_path}")
        tracks, video_frames, fps = run_sam3_on_video(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            video_path=video_path,
            conf_thresh=CONF_THRESH,
        )

        tree = build_cvat_xml(tracks)
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print(f"  Wrote XML: {xml_path}")

        create_annotated_video(tracks, video_frames, out_video_path, fps=fps)
        print()
    finally:
        if is_temp and video_path.exists():
            print(f"  Removing temporary MP4: {video_path.name}")
            video_path.unlink(missing_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def walk_dataset_and_process():
    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading SAM3 Video model on device={device}, dtype={dtype} ...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    print("Model loaded.\n")

    root = DATASET_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    print(f"Scanning dataset root: {root}")

    video_root = root / "video"
    if not video_root.exists():
        raise FileNotFoundError(f"'video' folder not found in {root}")

    print(f"=== Date directory (root): {root.name} ===")

    for cam_name in ["direita", "esquerda"]:
        cam_dir = video_root / cam_name
        if not cam_dir.exists():
            print(f"  [INFO] Camera dir not found: {cam_dir}")
            continue

        print(f"-- Camera: {cam_name} --")

        for videoid_dir in sorted(cam_dir.iterdir()):
            if not videoid_dir.is_dir():
                continue
            process_video_dir(model, processor, device, dtype, videoid_dir)


def main():
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    walk_dataset_and_process()


if __name__ == "__main__":
    main()
