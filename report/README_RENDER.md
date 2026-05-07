# Render instructions

Place these files in the existing `report/` directory, where `report/` is a sibling of `figures/`, `results/`, `logs/`, and `embeddings/`.

Recommended command from the project root:

```bash
quarto render report/writeup.qmd --to pdf
```

or from inside `report/`:

```bash
quarto render writeup.qmd --to pdf
```

The QMD computes the reported result values at render time via `report_metrics.py`. It reads:

- `../results/all_results.csv`
- `../results/alljoined_transfer.csv`
- `../logs/phase5/multi_subject_clip_chsubset32_seed42.status.json`
- `../embeddings/*.npy`
- selected `../results/**/*.npy` files for shape checks

The helper intentionally avoids pandas, SciPy, and mpmath as required dependencies. It uses `numpy` for loading `.npy` files and a small pure-Python Student-t calculation for paired-test p-values, so p-values should not render as `NA` on minimal Quarto installations.
