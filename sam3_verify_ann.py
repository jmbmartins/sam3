#!/usr/bin/env python3
from pathlib import Path

# ====== CONFIG (keep it consistent with your SAM3 script) ======
DATASET_ROOT = Path("/home/evox5090ia/sumasaojoao/2025-12-22_08-45-47").resolve()
OUTPUT_XML_NAME = "annotations.xml"
CAMERAS = ["direita", "esquerda"]
# ===============================================================


def main() -> None:
    video_root = DATASET_ROOT / "video"
    if not video_root.exists():
        raise FileNotFoundError(f"'video' folder not found: {video_root}")

    missing = []
    total = 0

    for cam in CAMERAS:
        cam_dir = video_root / cam
        if not cam_dir.exists():
            print(f"[WARN] Camera dir missing: {cam_dir}")
            continue

        for videoid_dir in sorted(cam_dir.iterdir()):
            if not videoid_dir.is_dir():
                continue

            total += 1
            xml_path = videoid_dir / OUTPUT_XML_NAME
            if not xml_path.exists():
                missing.append((cam, videoid_dir.name, str(videoid_dir)))

    print(f"Checked video folders: {total}")
    if not missing:
        print("[OK] All videoid folders have annotations.xml")
        return

    print(f"[MISSING] {len(missing)} folders without {OUTPUT_XML_NAME}:")
    for cam, vid, path in missing:
        print(f"  - {cam}/{vid}  -> {path}")


if __name__ == "__main__":
    main()
