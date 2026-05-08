# Render instructions

`report/writeup.qmd` is designed to live in `report/`, with `figures/`, `results/`, `logs/`, and `embeddings/` as sibling directories at the project root:

```text
CEREBRO/
  report/
    writeup.qmd
    references.bib
    report_metrics.py
  figures/
  results/
  logs/
  embeddings/
```

Render from the project root with:

```bash
quarto render report/writeup.qmd --to pdf
```

or from inside `report/` with:

```bash
quarto render writeup.qmd --to pdf
```

The QMD computes the reported result values at render time via `report_metrics.py`. The helper intentionally avoids pandas, SciPy, and mpmath as required dependencies. It uses `numpy` for loading `.npy` files and a small pure-Python Student-t calculation for paired-test p-values, so p-values should not render as `NA` on minimal Quarto installations.

## Minimal report artifacts

To populate only the artifacts needed for the report from a full `tribe-eeg/` folder, run this from the repo root:

```bash
python scripts/collect_report_artifacts.py --source /path/to/tribe-eeg --dest .
```

To preview what the script will copy without copying anything, run:

```bash
python scripts/collect_report_artifacts.py --source /path/to/tribe-eeg --dest . --list-only
```

The collector discovers figure dependencies by parsing `report/writeup.qmd`. It discovers result/log/embedding dependencies by importing `required_artifact_paths()` from `report/report_metrics.py`. This keeps the artifact list synchronized with the report and avoids maintaining a separate hardcoded list in the collector script.
