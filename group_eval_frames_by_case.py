import csv
import shutil
from pathlib import Path


ROOT = Path("/home/evox5090ia/Downloads/helmet_detection/eval_frames")
CASE_ROOT = ROOT / "by_case"


def infer_case(frame_path: Path) -> str:
    name = frame_path.name

    if "traseira_11" in name or "traseira_9" in name:
        return "partial_operator_top_edge"

    if "traseira_3" in name:
        rep_index = int(name.split("_rep_")[1].split("_")[0])
        if rep_index <= 6:
            return "no_operator_low_clutter"
        return "no_operator_high_clutter"

    return "uncategorized"


def main() -> None:
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []

    frame_paths = sorted(
        path
        for path in ROOT.glob("*/*.jpg")
        if path.is_file()
    )

    for frame_path in frame_paths:
        case_name = infer_case(frame_path)
        case_dir = CASE_ROOT / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        destination = case_dir / frame_path.name
        shutil.copy2(frame_path, destination)
        manifest_rows.append(
            {
                "frame": str(frame_path),
                "case": case_name,
                "destination": str(destination),
            }
        )

    manifest_path = CASE_ROOT / "case_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "case", "destination"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Grouped {len(frame_paths)} frames into {CASE_ROOT}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
