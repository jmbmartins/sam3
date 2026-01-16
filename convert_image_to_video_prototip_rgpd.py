from pathlib import Path
import re
import numpy as np
from PIL import Image
import imageio.v2 as imageio

# ============== CONFIG ==============
IMAGES_DIR = Path("/home/evox5090ia/sumasaojoao/2025-12-22_08-45-47_for_ann/joaojoaq/detections_blackout")
OUT_DIR = IMAGES_DIR.parent / "videos_blackout"

FPS = 10  # change if needed
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# ====================================


def split_video_and_frame(stem: str):
    """
    Given "videoid_framenumber", returns (videoid, framenumber_str).
    Uses rsplit so videoid can contain underscores.
    """
    if "_" not in stem:
        return None, None
    video_id, frame_part = stem.rsplit("_", 1)
    return video_id, frame_part


def frame_sort_key(frame_part: str):
    """
    Prefer numeric sort if frame_part contains a number, else string sort.
    """
    m = re.search(r"\d+", frame_part)
    if m:
        return (0, int(m.group()))
    return (1, frame_part)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in: {IMAGES_DIR}")
        return

    # Group paths by video_id
    groups = {}
    for p in image_paths:
        video_id, frame_part = split_video_and_frame(p.stem)
        if video_id is None:
            continue
        groups.setdefault(video_id, []).append((frame_part, p))

    if not groups:
        print("No valid files matching videoid_framenumber.*")
        return

    for video_id, items in sorted(groups.items()):
        # Sort frames
        items_sorted = sorted(items, key=lambda x: frame_sort_key(x[0]))
        frame_files = [p for _, p in items_sorted]

        out_path = OUT_DIR / f"{video_id}.mp4"
        print(f"[{video_id}] {len(frame_files)} frames -> {out_path}")

        # Read first frame to get size
        first = np.array(Image.open(frame_files[0]).convert("RGB"))
        h, w = first.shape[:2]

        # Write MP4
        writer = imageio.get_writer(
            out_path,
            fps=FPS,
            codec="libx264",
            quality=8,          # 0(best)-10(worst) depending on backend; good default
            pixelformat="yuv420p"  # best compatibility
        )

        try:
            # Ensure consistent size; resize if needed
            for fp in frame_files:
                img = Image.open(fp).convert("RGB")
                if img.size != (w, h):
                    img = img.resize((w, h), Image.BILINEAR)
                writer.append_data(np.array(img))
        finally:
            writer.close()

        print(f"[{video_id}] done")

    print("\nAll videos created.")


if __name__ == "__main__":
    main()
