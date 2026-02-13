#!/usr/bin/env python3
import subprocess
from pathlib import Path
import sys

# ---- HARD-CODED INPUT MKV PATH ----
INPUT_MKV = Path(
    "/home/evox5090ia/Downloads/2025-12-23_08-56-33/video/traseira/2025-12-23_08-56-33_traseira_39.mkv"
)
# -----------------------------------


def main():
    if not INPUT_MKV.exists():
        print(f"Error: input file not found: {INPUT_MKV}")
        sys.exit(1)

    # Same name, .mp4 extension
    output_mp4 = "/home/evox5090ia/Downloads/2025-12-23_08-56-33_traseira_39.mp4"

    # ffmpeg remux (no re-encode, fast, no quality loss)
    cmd = [
        "ffmpeg",
        "-y",              # overwrite existing output without asking
        "-i", str(INPUT_MKV),
        "-c", "copy",      # copy streams (no re-encoding)
        str(output_mp4),
    ]

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"Done! Output: {output_mp4}")
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed with code:", e.returncode)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
