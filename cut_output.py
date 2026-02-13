#!/usr/bin/env python3
import subprocess
from pathlib import Path

# ===== HARDCODED SETTINGS =====
IN_VIDEO = "/home/evox5090ia/Downloads/output_FIXED.mp4"
OUT_VIDEO = "/home/evox5090ia/Downloads/output_FIXED_clip.mp4"

START = "00:01:10"     # HH:MM:SS (ou HH:MM:SS.mmm)
END   = "00:01:40"     # ou deixa END=None e usa DURATION
DURATION = None        # ex: "00:00:30" ou "30" (segundos) se END=None

EXACT_CUT = False      # False = rápido (stream copy), True = exato (re-encode)
# ==============================


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    inp = Path(IN_VIDEO)
    out = Path(OUT_VIDEO)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    if END is None and DURATION is None:
        raise SystemExit("Set END or DURATION (one of them must be not None).")

    if EXACT_CUT:
        # Frame-perfect cut (re-encode) - slower but exact
        cmd = ["ffmpeg", "-hide_banner", "-y", "-ss", START, "-i", str(inp)]
        if END is not None:
            cmd += ["-to", END]
        else:
            cmd += ["-t", str(DURATION)]
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out),
        ]
    else:
        # Fast cut (stream copy) - very fast, may not be frame-perfect
        cmd = ["ffmpeg", "-hide_banner", "-y", "-ss", START, "-i", str(inp)]
        if END is not None:
            cmd += ["-to", END]
        else:
            cmd += ["-t", str(DURATION)]
        cmd += ["-c", "copy", "-movflags", "+faststart", str(out)]

    run(cmd)
    print(f"✓ Wrote: {out}")


if __name__ == "__main__":
    main()
