import csv
import subprocess
import tempfile
from pathlib import Path


INPUT_DIR = Path("/home/evox5090ia/Downloads/helmet_detection")
OUTPUT_ROOT = INPUT_DIR / "operator_case_benchmark"

CASE_SPECS = [
    {
        "case": "q1_no_operator_controls",
        "items": [
            ("2026-03-03_15-25-16_traseira_3.mkv", 85.0, "No visible operator; control for false alarms."),
            ("2026-03-03_15-25-16_traseira_3.mkv", 88.0, "No visible operator; control for false alarms."),
            ("2026-03-03_15-25-16_traseira_3.mkv", 90.0, "No visible operator; control for false alarms."),
        ],
    },
    {
        "case": "q2_clear_helmet_present",
        "items": [
            ("2026-03-03_15-25-16_traseira_11.mkv", 13.0, "Left operator shows clear helmet evidence."),
            ("2026-03-03_15-25-16_traseira_11.mkv", 26.0, "Left operator shows clear helmet evidence."),
            ("2026-03-03_15-25-16_traseira_9.mkv", 15.0, "Left operator shows clear helmet evidence."),
            ("2026-03-03_15-25-16_traseira_9.mkv", 53.0, "Left operator still shows clear helmet evidence."),
        ],
    },
    {
        "case": "q3_two_operators_simultaneous",
        "items": [
            ("2026-03-03_15-25-16_traseira_11.mkv", 13.0, "Two operators at the top border; mixed visibility."),
            ("2026-03-03_15-25-16_traseira_11.mkv", 26.0, "Two operators at the top border; one clearer than the other."),
            ("2026-03-03_15-25-16_traseira_9.mkv", 15.0, "Two operators visible at once with one clearer left-side helmet."),
            ("2026-03-03_15-25-16_traseira_9.mkv", 53.0, "Two operators visible at once; useful for multi-person association."),
        ],
    },
    {
        "case": "q4_head_partial_top_border",
        "items": [
            ("2026-03-03_15-25-16_traseira_11.mkv", 55.0, "Top-edge operator visibility with partial head and drift risk."),
            ("2026-03-03_15-25-16_traseira_9.mkv", 55.0, "Top-edge partial head visibility; hard for helmet absence reasoning."),
            ("2026-03-03_15-25-16_traseira_3.mkv", 92.0, "Left operator only partially visible at the top border."),
        ],
    },
    {
        "case": "q5_hard_ambiguous_should_be_unknown",
        "items": [
            ("2026-03-03_15-25-16_traseira_3.mkv", 95.0, "Single top-edge operator with weak head evidence; should stay conservative."),
            ("2026-03-03_15-25-16_traseira_3.mkv", 96.0, "Two partial top-edge operators; should not confidently trigger no-helmet."),
            ("2026-03-03_15-25-16_traseira_11.mkv", 55.0, "Hard ambiguous frame with noisy associations."),
            ("2026-03-03_15-25-16_traseira_9.mkv", 55.0, "Mixed evidence; a conservative system should often stay unknown."),
        ],
    },
]


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
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        converted = {}

        for spec in CASE_SPECS:
            case_dir = OUTPUT_ROOT / spec["case"]
            case_dir.mkdir(parents=True, exist_ok=True)

            for video_name, timestamp_s, note in spec["items"]:
                source_path = INPUT_DIR / video_name
                processing_path = converted.get(video_name)
                if processing_path is None:
                    processing_path = convert_mkv_to_mp4(source_path, temp_dir)
                    converted[video_name] = processing_path

                frame_name = f"{source_path.stem}_{timestamp_s:07.3f}s.jpg"
                output_path = case_dir / frame_name
                extract_frame(processing_path, timestamp_s, output_path)
                manifest_rows.append(
                    {
                        "case": spec["case"],
                        "video": video_name,
                        "timestamp_s": f"{timestamp_s:.3f}",
                        "frame": str(output_path),
                        "note": note,
                    }
                )

    manifest_path = OUTPUT_ROOT / "benchmark_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "video", "timestamp_s", "frame", "note"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Built benchmark folders at {OUTPUT_ROOT}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
