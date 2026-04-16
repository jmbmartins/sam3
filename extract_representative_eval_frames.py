import csv
import json
import subprocess
import tempfile
from pathlib import Path


INPUT_DIR = Path("/home/evox5090ia/Downloads/helmet_detection")
OUTPUT_DIR = INPUT_DIR / "eval_frames"
FRAMES_PER_VIDEO = 12
SUPPORTED_SUFFIXES = {".mkv"}


def ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    return float(payload["format"]["duration"])


def representative_timestamps(duration_s: float, count: int) -> list[float]:
    if count <= 1:
        return [duration_s * 0.5]

    start_ratio = 0.08
    end_ratio = 0.92
    start_t = duration_s * start_ratio
    end_t = duration_s * end_ratio
    step = (end_t - start_t) / (count - 1)
    return [start_t + i * step for i in range(count)]


def convert_mkv_to_mp4(video_path: Path, temp_dir: Path) -> Path:
    temp_mp4 = temp_dir / f"{video_path.stem}_temp.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(temp_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return temp_mp4


def extract_frame(video_path: Path, timestamp_s: float, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp_s:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "manifest.csv"

    rows: list[dict[str, str]] = []
    video_paths = sorted(
        path for path in INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    for video_path in video_paths:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            processing_path = video_path
            if video_path.suffix.lower() == ".mkv":
                processing_path = convert_mkv_to_mp4(video_path, temp_dir)

            duration_s = ffprobe_duration(processing_path)
            timestamps = representative_timestamps(duration_s, FRAMES_PER_VIDEO)
            video_output_dir = OUTPUT_DIR / video_path.stem
            video_output_dir.mkdir(parents=True, exist_ok=True)

            for index, timestamp_s in enumerate(timestamps, start=1):
                frame_name = f"{video_path.stem}_rep_{index:02d}_{timestamp_s:07.3f}s.jpg"
                frame_path = video_output_dir / frame_name
                extract_frame(processing_path, timestamp_s, frame_path)
                rows.append(
                    {
                        "video": video_path.name,
                        "frame": str(frame_path),
                        "timestamp_s": f"{timestamp_s:.3f}",
                    }
                )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "frame", "timestamp_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} frames to {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
