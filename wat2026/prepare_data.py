from __future__ import annotations

import argparse
import gzip
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "ShivRamSaud/HINDI-BENGALI-MALAYALAM-ODIA-VISUAL-GENOME"
DEFAULT_ZIP = "hindi-visual-genome-11.zip"


def count_rows(path: Path) -> int:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the WAT 2026 Visual Genome data"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--zip-name", default=DEFAULT_ZIP)
    parser.add_argument("--data-root", default="data/visual-genome")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = snapshot_download(
        args.repo_id,
        repo_type="dataset",
        allow_patterns=["*.txt", "*.txt.gz", "*.tsv.gz", args.zip_name],
    )
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(snapshot)
    for path in list(snapshot_path.glob("*.txt*")) + list(
        snapshot_path.glob("*.tsv*")
    ):
        shutil.copy(path, root / path.name)
    marker = root / ".images_unzipped"
    if not marker.exists():
        with zipfile.ZipFile(snapshot_path / args.zip_name) as archive:
            archive.extractall(root)
        marker.touch()
    for lang, pattern in {
        "hi": "hindi-visual-genome-*.txt",
        "bn": "bengali-visual-genome-*.txt",
        "ml": "malayalam-visual-genome-*.txt.gz",
        "or": "odia-visual-genome-*.tsv.gz",
    }.items():
        for path in sorted(root.glob(pattern)):
            if "train" in path.name or "dev" in path.name:
                print(f"{lang} {path.name} rows={count_rows(path)}")
    print(f"data_root={root}")


if __name__ == "__main__":
    main()
