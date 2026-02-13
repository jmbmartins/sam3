import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import imageio.v3 as iio

import supervision as sv
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor


# ================= CONFIG =================

INPUT_VIDEO_DIR = "/home/evox5090ia/Downloads/2025-12-23_08-56-33/video/traseira/"
OUTPUT_DIR = "/home/evox5090ia/experiments/traseira_baldeacao/contaminacao/2025-12-23_08-56-33_papamarelo"

CONF_THRESH = 0.50
OUTPUT_FPS = 10
FRAME_STRIDE = 5   # process 1 frame every 5 (speed boost)

CONTAMINATION_CLASSES = [
    "glass bottle",
    "glass jar",
    "cardboard box",
    "ceramic plate",
    "pan or pot",
    "tool",
    "bucket",
    "broom or mop",
    "appliance",
    "cutlery",
    "cd or dvd"
]

COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff",
    "#b266ff", "#9999ff", "#3399ff", "#66ffff",
    "#33ff99", "#66ff66", "#99ff00"
])

# =========================================


def safe_load_video(video_path: Path, stride: int) -> Optional[List[np.ndarray]]:
    """
    Loader robusto via imageio. Já aplica stride no streaming para não estourar RAM.
    """
    try:
        frames: List[np.ndarray] = []
        for idx, frame in enumerate(iio.imiter(video_path)):
            if stride > 1 and (idx % stride != 0):
                continue
            frames.append(frame)

        if len(frames) == 0:
            raise ValueError("No frames decoded")

        return frames
    except Exception as e:
        print(f"❌ Failed to load {video_path.name}: {e}")
        return None


def annotate(image: Image.Image, detections: sv.Detections) -> Image.Image:
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)

    box_annotator = sv.BoxAnnotator(color=COLOR, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=COLOR,
        text_scale=text_scale,
        text_padding=6,
        text_color=sv.Color.BLACK,
        text_thickness=1,
    )

    labels = [
        f"{name} {conf:.2f}"
        for name, conf in zip(detections.data["class_name"], detections.confidence)
    ]

    img = image.copy()
    img = box_annotator.annotate(img, detections)
    img = label_annotator.annotate(img, detections, labels)
    return img


def _pick_class_id_field(frame_outputs: Dict[str, Any]) -> Optional[str]:
    """
    Encontra automaticamente o campo mais provável que representa "ID de classe/prompt".
    Dependendo da versão, pode aparecer como class_ids, labels, text_ids, prompt_ids, etc.
    """
    if frame_outputs is None:
        return None

    candidates = [
        "class_ids",
        "class_id",
        "labels",
        "label_ids",
        "text_ids",
        "prompt_ids",
        "category_ids",
    ]
    for k in candidates:
        if k in frame_outputs:
            return k
    return None


def sam3_outputs_to_detections(
    frame_outputs: Dict[str, Any],
    conf_thresh: float
) -> Optional[sv.Detections]:
    if frame_outputs is None:
        return None

    # Obrigatórios (típicos)
    boxes = frame_outputs.get("boxes", None)
    scores = frame_outputs.get("scores", None)
    object_ids = frame_outputs.get("object_ids", None)  # track ids (não é class id)

    if boxes is None or scores is None or object_ids is None:
        # Se a tua versão devolver chaves diferentes, imprime as keys para ajustares
        print(f"⚠️ frame_outputs keys inesperadas: {list(frame_outputs.keys())}")
        return None

    keep = scores > conf_thresh
    if int(keep.sum().item()) == 0:
        return None

    boxes_np = boxes[keep].detach().cpu().numpy().astype(np.float32)
    scores_np = scores[keep].detach().cpu().numpy().astype(np.float32)
    obj_ids_np = object_ids[keep].detach().cpu().numpy().astype(np.int32)

    # Tentar obter IDs de classe/prompt a partir do campo correto
    class_field = _pick_class_id_field(frame_outputs)
    class_ids_np: Optional[np.ndarray] = None

    if class_field is not None:
        raw = frame_outputs[class_field]
        try:
            class_ids_np = raw[keep].detach().cpu().numpy().astype(np.int32)
        except Exception:
            # Alguns formatos não usam máscara keep no mesmo tensor; tenta converter sem mask e alinhar por tamanho
            try:
                raw_np = raw.detach().cpu().numpy().astype(np.int32)
                if raw_np.shape[0] == obj_ids_np.shape[0]:
                    class_ids_np = raw_np
                else:
                    class_ids_np = None
            except Exception:
                class_ids_np = None

    # Se não houver class_ids válidos, fallback seguro (sem crash)
    if class_ids_np is None:
        # NÃO é ideal (labels podem ficar errados), mas mantém pipeline vivo
        # e evita IndexError. Também avisa uma vez.
        class_ids_np = np.full((obj_ids_np.shape[0],), -1, dtype=np.int32)

    # Map de nomes de classe
    class_names: List[str] = []
    for cid in class_ids_np.tolist():
        if 0 <= cid < len(CONTAMINATION_CLASSES):
            class_names.append(CONTAMINATION_CLASSES[cid])
        else:
            class_names.append("unknown")

    detections = sv.Detections(
        xyxy=boxes_np,
        confidence=scores_np,
        class_id=class_ids_np,
    )

    # track id separado (se a tua versão do supervision suportar)
    try:
        detections.tracker_id = obj_ids_np
    except Exception:
        detections.data["tracker_id"] = obj_ids_np

    detections.data["class_name"] = np.array(class_names, dtype=object)
    return detections


def process_video(
    model,
    processor,
    video_frames: List[np.ndarray],
    device,
    dtype
) -> Tuple[List[np.ndarray], bool]:

    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype
    )

    # Adiciona prompts (ordem define índices 0..N-1)
    for cls in CONTAMINATION_CLASSES:
        inference_session = processor.add_text_prompt(
            inference_session=inference_session,
            text=cls
        )

    outputs_per_frame: Dict[int, Dict[str, Any]] = {}
    for out in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=len(video_frames),
    ):
        outputs_per_frame[out.frame_idx] = processor.postprocess_outputs(inference_session, out)

    annotated_frames: List[np.ndarray] = []
    has_detections = False

    for idx, frame in enumerate(video_frames):
        frame_pil = Image.fromarray(frame).convert("RGB")

        detections = sam3_outputs_to_detections(outputs_per_frame.get(idx), CONF_THRESH)
        if detections is not None:
            # (opcional) só considerar se houver classe conhecida
            if np.any(detections.class_id >= 0):
                has_detections = True
                frame_pil = annotate(frame_pil, detections)

        annotated_frames.append(np.array(frame_pil))

    # cleanup
    del inference_session
    del outputs_per_frame
    torch.cuda.empty_cache()
    gc.collect()

    return annotated_frames, has_detections


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"🖥️ Device: {device}")

    print("📥 Loading SAM3 Video model...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    print("✅ Model loaded")

    video_paths = sorted(Path(INPUT_VIDEO_DIR).glob("*.mkv"))
    print(f"📂 Found {len(video_paths)} videos")

    for video_path in video_paths:
        print(f"\n🎥 Processing: {video_path.name}")

        frames = safe_load_video(video_path, stride=FRAME_STRIDE)
        if frames is None or len(frames) < 5:
            print("⚠️ Skipping video (broken or too short)")
            continue

        try:
            annotated_frames, has_detections = process_video(model, processor, frames, device, dtype)
        except RuntimeError as e:
            print(f"🔥 Runtime error on {video_path.name}: {e}")
            torch.cuda.empty_cache()
            continue

        if not has_detections:
            print("🚫 No contamination detected — skipping save")
            continue

        out_path = Path(OUTPUT_DIR) / f"{video_path.stem}_CONTAMINATION.mp4"
        print(f"💾 Saving: {out_path.name}")

        # macro_block_size=1 evita resize 1080->1088 (pode reduzir compatibilidade em players antigos)
        iio.imwrite(
            out_path,
            annotated_frames,
            fps=OUTPUT_FPS,
            codec="libx264",
            macro_block_size=1
        )

        print("✅ Saved")

    print("\n🎉 DONE — contamination scan complete")


if __name__ == "__main__":
    main()
