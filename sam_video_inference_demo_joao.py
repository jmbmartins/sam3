import torch
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image

import cv2
import supervision as sv
from accelerate import Accelerator

from transformers import Sam3VideoModel, Sam3VideoProcessor

import imageio.v3 as iio
import gc


# =========================
# CONFIG
# =========================
INPUT_FOLDER = "/home/evox5090ia/Downloads/videos_demonstracao/videos_ano"
OUTPUT_FOLDER = "/home/evox5090ia/Downloads/videos_demonstracao/videos_ano_sam"

CHUNK_SIZE = 25
CONF_THRESH = 0.05
OUTPUT_FPS = 25


# =========================
# PROMPTS
# =========================
PROMPTS = [
    ("yellow container", "contentor_plastico_metal"),
    ("blue container", "contentor_papel_cartao"),
    ("green container", "contentor_vidro"),
    ("black bin bag", "saco_de_residuo"),
    ("cardboard box", "caixa_de_cartao"),
    ("graffiti", "grafite")
]


# =========================
# COLOR PALETTE
# =========================
COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff",
    "#b266ff", "#9999ff", "#3399ff", "#66ffff", "#33ff99",
    "#66ff66", "#99ff00"
])


# =========================
# GLOBAL TRACK STATE
# =========================
global_tracks = []
next_track_id = 0


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


def assign_track_ids(detections):
    global global_tracks, next_track_id

    assigned = []

    if detections is None or len(detections) == 0:
        return []

    for box in detections.xyxy:

        best_id = None
        best_score = 0

        for t in global_tracks:
            score = iou(box, t["box"])
            if score > best_score and score > 0.3:
                best_score = score
                best_id = t["id"]

        if best_id is None:
            best_id = next_track_id
            next_track_id += 1

        assigned.append(best_id)

        found = False
        for t in global_tracks:
            if t["id"] == best_id:
                t["box"] = box
                t["missed"] = 0
                found = True

        if not found:
            global_tracks.append({"id": best_id, "box": box, "missed": 0})

    return assigned


# =========================
# DETECTIONS
# =========================
def sam3_video_frame_to_detections(frame_outputs: Dict[str, Any], conf_thresh=0.05):

    if frame_outputs is None:
        return None

    boxes = frame_outputs["boxes"]
    scores = frame_outputs["scores"]
    masks = frame_outputs["masks"]
    object_ids = frame_outputs["object_ids"]

    keep = scores > conf_thresh
    if keep.sum() == 0:
        return None

    return sv.Detections(
        xyxy=boxes[keep].cpu().numpy().astype(np.float32),
        mask=(masks[keep].cpu().numpy() > 0.5),
        confidence=scores[keep].cpu().numpy().astype(np.float32),
        class_id=np.zeros(len(boxes[keep]), dtype=np.int32),
    )


# =========================
# ANNOTATION
# =========================
def annotate(image, detections, track_ids):

    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)

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

    annotated = image.copy()
    annotated = box_annotator.annotate(annotated, detections)

    labels = []
    for tid, conf in zip(track_ids, detections.confidence):
        labels.append(f"ID:{tid} contentor {conf:.2f}")

    return label_annotator.annotate(annotated, detections, labels)


# =========================
# PROCESS VIDEO
# =========================
def process_video(video_path, model, processor, device, dtype):

    print(f"\nProcessing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    OUTPUT_FOLDER_PATH = Path(OUTPUT_FOLDER)
    OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    out_file = OUTPUT_FOLDER_PATH / f"{video_path.stem}_sam.mp4"

    buffer = []
    output_frames = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer.append(frame)

        if len(buffer) == CHUNK_SIZE:

            inference_session = processor.init_video_session(
                video=buffer,
                inference_device=device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=dtype
            )

            for prompt, _ in PROMPTS:
                inference_session = processor.add_text_prompt(
                    inference_session=inference_session,
                    text=prompt,
                )

            outputs = {}

            for model_outputs in model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=len(buffer),
            ):
                outputs[model_outputs.frame_idx] = processor.postprocess_outputs(
                    inference_session, model_outputs
                )

            for i, frame in enumerate(buffer):

                pil = Image.fromarray(frame)
                out = outputs.get(i)

                if out is None:
                    output_frames.append(np.array(pil))
                    continue

                det = sam3_video_frame_to_detections(out, CONF_THRESH)

                if det is None:
                    output_frames.append(np.array(pil))
                    continue

                track_ids = assign_track_ids(det)

                output_frames.append(np.array(annotate(pil, det, track_ids)))

            # cleanup
            buffer = []
            del inference_session
            del outputs
            gc.collect()
            torch.cuda.empty_cache()

    # leftover frames
    if buffer:

        inference_session = processor.init_video_session(
            video=buffer,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=dtype
        )

        for prompt, _ in PROMPTS:
            inference_session = processor.add_text_prompt(inference_session, prompt)

        outputs = {}

        for model_outputs in model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=len(buffer),
        ):
            outputs[model_outputs.frame_idx] = processor.postprocess_outputs(
                inference_session, model_outputs
            )

        for i, frame in enumerate(buffer):

            pil = Image.fromarray(frame)
            out = outputs.get(i)

            if out is None:
                output_frames.append(np.array(pil))
                continue

            det = sam3_video_frame_to_detections(out, CONF_THRESH)

            if det is None:
                output_frames.append(np.array(pil))
                continue

            track_ids = assign_track_ids(det)

            output_frames.append(np.array(annotate(pil, det, track_ids)))

    cap.release()

    iio.imwrite(out_file, output_frames, fps=OUTPUT_FPS, codec="libx264")

    print(f"Saved: {out_file}")


# =========================
# MAIN
# =========================
def main():

    input_dir = Path(INPUT_FOLDER)
    videos = sorted(input_dir.glob("*.mp4"))

    accelerator = Accelerator()
    device = accelerator.device

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("Loading SAM3...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    print(f"Found {len(videos)} videos")

    for v in videos:
        process_video(v, model, processor, device, dtype)


if __name__ == "__main__":
    main()