#!/usr/bin/env python3
"""
Copy only the artifacts needed to render report/writeup.qmd.

The script discovers:
1. Figure/image files referenced directly by report/writeup.qmd.
2. Data/log/embedding files declared by report/report_metrics.py.

Usage:
    python scripts/collect_report_artifacts.py --source /path/to/full/tribe-eeg --dest .
    python scripts/collect_report_artifacts.py --source /path/to/full/tribe-eeg --dest . --list-only
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse


QMD_IMAGE_PATTERNS = [
    # Markdown images: ![caption](../figures/foo.png){#fig-id width="70%"}
    re.compile(r"!\[[^\]]*\]\(([^)]+)\)"),
    # Quarto / knitr-style includes, in case the report later switches format.
    re.compile(r"include_graphics\([\"']([^\"']+)[\"']\)"),
    # Conservative fallback for quoted figure paths in code chunks.
    re.compile(r"[\"']((?:\.\./)?figures/[^\"']+\.(?:png|pdf|jpg|jpeg|svg))[\"']"),
]


def is_local_path(path_str: str) -> bool:
    parsed = urlparse(path_str)
    return parsed.scheme == "" and not path_str.startswith("#")


def normalize_qmd_reference(qmd_path: Path, path_str: str, project_root: Path) -> Path:
    """Return a project-root-relative path for a local QMD reference."""
    # Strip optional Quarto attributes if they were captured accidentally.
    # Example: ../figures/foo.png{#fig-x width="70%"}
    path_str = path_str.split("{", 1)[0].strip()
    abs_path = (qmd_path.parent / path_str).resolve()
    return abs_path.relative_to(project_root.resolve())


def discover_qmd_artifacts(qmd_path: Path, project_root: Path) -> list[Path]:
    text = qmd_path.read_text(encoding="utf-8")
    discovered: set[Path] = set()

    for pattern in QMD_IMAGE_PATTERNS:
        for match in pattern.finditer(text):
            ref = match.group(1).strip()
            if not is_local_path(ref):
                continue
            try:
                discovered.add(normalize_qmd_reference(qmd_path, ref, project_root))
            except ValueError:
                # Ignore references outside the project root.
                pass

    return sorted(discovered)


def load_metrics_artifacts(report_metrics_path: Path, project_root: Path) -> list[Path]:
    spec = importlib.util.spec_from_file_location("report_metrics", report_metrics_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {report_metrics_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "required_artifact_paths"):
        raise RuntimeError(
            "report/report_metrics.py must define required_artifact_paths()."
        )

    relative_paths: set[Path] = set()
    for path in module.required_artifact_paths():
        path = Path(path)
        if path.is_absolute():
            relative_paths.add(path.resolve().relative_to(project_root.resolve()))
        else:
            relative_paths.add(path)

    return sorted(relative_paths)


def copy_files(required_files: list[Path], source_root: Path, dest_root: Path) -> None:
    missing: list[Path] = []

    for relative in required_files:
        src = source_root / relative
        dst = dest_root / relative

        if not src.exists():
            missing.append(relative)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {relative}")

    if missing:
        print("\nMissing required files:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to full tribe-eeg folder")
    parser.add_argument("--dest", default=".", help="Repo root destination")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print discovered files without copying them.",
    )
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    dest_root = Path(args.dest).expanduser().resolve()

    qmd_path = dest_root / "report" / "writeup.qmd"
    report_metrics_path = dest_root / "report" / "report_metrics.py"

    if not qmd_path.exists():
        raise SystemExit(f"Could not find {qmd_path}")
    if not report_metrics_path.exists():
        raise SystemExit(f"Could not find {report_metrics_path}")

    qmd_artifacts = discover_qmd_artifacts(qmd_path, dest_root)
    metrics_artifacts = load_metrics_artifacts(report_metrics_path, dest_root)
    required_files = sorted(set(qmd_artifacts + metrics_artifacts))

    print("Required report artifacts:")
    for path in required_files:
        print(f"  - {path}")

    if args.list_only:
        return

    print()
    copy_files(required_files, source_root, dest_root)
    print("\nDone. Minimal report artifacts copied.")


if __name__ == "__main__":
    main()
