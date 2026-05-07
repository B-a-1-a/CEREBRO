#!/usr/bin/env python3
"""
Copy only the artifacts needed to render report/writeup.qmd.

Usage:
    python scripts/collect_report_artifacts.py --source /path/to/tribe-eeg --dest .
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REQUIRED_FILES = [
    # Figures used by the report
    "figures/headline_boxplot.png",
    "figures/headline_boxplot.pdf",
    "figures/channel_time_heatmap.png",
    "figures/channel_time_heatmap.pdf",
    "figures/embedding_ablation.png",
    "figures/embedding_ablation.pdf",
    "figures/alljoined_transfer.png",
    "figures/alljoined_transfer.pdf",

    # Main result summaries
    "results/all_results.csv",
    "results/alljoined_transfer.csv",

    # Logs / masks used by report_metrics.py
    "logs/channel_mask_28.npy",
    "logs/phase5/multi_subject_clip_chsubset32_seed42.status.json",

    # Embedding matrices used only for shape checks in the report
    "embeddings/clip_train.npy",
    "embeddings/clip_test.npy",
    "embeddings/dinov2_train.npy",
    "embeddings/dinov2_test.npy",
]


def copy_required_files(source_root: Path, dest_root: Path) -> None:
    missing: list[str] = []

    for relative in REQUIRED_FILES:
        src = source_root / relative
        dst = dest_root / relative

        if not src.exists():
            missing.append(relative)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {relative}")

    if missing:
        print("\nMissing files:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\nDone. Minimal report artifacts copied.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to full tribe-eeg folder")
    parser.add_argument("--dest", default=".", help="Repo root destination")
    args = parser.parse_args()

    copy_required_files(Path(args.source).expanduser().resolve(), Path(args.dest).expanduser().resolve())


if __name__ == "__main__":
    main()