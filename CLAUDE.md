# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CEREBRO tests whether Meta's TRIBE v2 architectural insight — subject-conditioned transformers trained jointly across participants — transfers from slow fMRI to fast event-related EEG. The novel adaptation is an **RSVP context window** (±5 adjacent image embeddings) replacing TRIBE's 100-second fMRI sliding window.

**Datasets:** THINGS-EEG2 (10 subjects, ~820k trials, 63 channels) and Alljoined-1.6M (20 subjects, ~1.6M trials, 32 channels).

**Three models under comparison:**
1. Per-subject linear baseline (ridge/linear, trained separately per subject)
2. Per-subject transformer (shared architecture, trained independently per subject)
3. Multi-subject transformer + subject block (trained jointly across all subjects)

## Development Environment

This is a **Google Colab-first project**. All heavy computation runs in Colab notebooks. All persistent artifacts (data, embeddings, checkpoints, results, figures) live on Google Drive at `/content/drive/MyDrive/tribe-eeg/`. The local repo holds only source code, configs, and notebooks.

### Install dependencies
```bash
pip install -r requirements.txt
```

Key packages: `torch`, `torchvision`, `transformers==4.44.0`, `mne==1.7.1`, `einops`, `h5py`, `pyyaml`, `scikit-learn`.

### No test suite
There is no `pytest` suite. Correctness is validated through in-notebook sanity checks at each phase (see Phases 1c, 2c, 4 in `ROADMAP.md`).

## Execution Pipeline

The 6 notebooks run **sequentially** through Phases 0-8:

| Notebook | Phases | Key output |
|---|---|---|
| `01_data_download.ipynb` | 0–2 | Averaged EEG `.npz` files, image filename manifests |
| `02_embedding_extraction.ipynb` | 3 | CLIP/DINOv2 memmaps (~150 MB) |
| `03_baselines.ipynb` | 4 | Ridge sanity check; expect mean Pearson R ≈ 0.10–0.25 |
| `04_transformer_training.ipynb` | 5 | Trained checkpoints for all three models |
| `05_evaluation.ipynb` | 6 | Box plot, channel×time heatmap, embedding ablation, results table |
| `06_alljoined_transfer.ipynb` | 7 | Zero-shot and fine-tuned Alljoined transfer results |

## Architecture

### Data flow
`ThingsEEG2Dataset` (fully implemented in `src/data/things_eeg2.py`) loads per-subject averaged EEG from `.npz` and embeddings from numpy arrays, constructs RSVP context windows (±5 neighbors, zero-padded at boundaries), and returns `(image_emb, context_embs, subject_idx, eeg)`.

### Model architecture (stubs in `src/models/`)
- **`SubjectBlock`**: Parameter matrix `(n_subjects, d_model, C×T)` indexed by `subject_id` — the subject-conditional output projection layer.
- **`MultiSubjectTransformer`**: Shared 4-layer, 6-head transformer (d_model=384, ffn_dim=1536) followed by `SubjectBlock` for subject-conditional decoding. Trained jointly on all subjects.
- **`PerSubjectTransformer`**: Same architecture minus `SubjectBlock`, trained independently per subject.

### Training conventions (from `configs/`)
- Optimizer: AdamW, lr=1e-4, weight_decay=0.01
- Schedule: Cosine with 5% linear warmup
- Batch size: 64, epochs: 30, early stopping patience: 5
- Random seeds: 42 (primary), 1337, 2026
- Primary embedding: CLIP ViT-L/14 (d_emb=768); DINOv2-Large (d_emb=1024) for ablation

### Loss and metrics
- Training loss: MSE on EEG predictions (`src/training/losses.py`)
- Eval metric: Pearson correlation per channel×timepoint, then averaged (`src/eval/metrics.py`)

## Google Drive Layout

```
/content/drive/MyDrive/tribe-eeg/
├── raw/thingseeg2_preproc/      # sub-01 through sub-10 (preprocessed EEG)
├── raw/thingseeg2_metadata/     # training_images/ (1654 concepts), test_images/ (200)
├── processed/                   # eeg_train_avg_sub-*.npz, image filename JSONs
├── embeddings/                  # clip_vitl14_{train,test}.npy, dinov2_large_{train,test}.npy
├── checkpoints/                 # linear_baseline/, per_subject_transformer/, multi_subject_transformer/
├── logs/                        # JSONL training logs, env.json, data inventory
├── results/                     # ridge results, final eval CSV
└── figures/                     # PNG/PDF plots
```

## Implementation Status

| Component | Status |
|---|---|
| `src/data/things_eeg2.py` | Complete |
| `src/data/alljoined.py` | Stub |
| `src/models/` (all) | Stubs |
| `src/training/` | Stubs (partial losses) |
| `src/eval/metrics.py` | Stubs |
| `scripts/` | Stubs |
| Notebooks 01–02 | Implemented |
| Notebooks 03–06 | Stubs |

## Operational Rules (from `specs/CLAUDE_CODE_SPEC.md`)

- Always check for existing artifacts before recomputing — never overwrite without `--force`.
- Checkpoint aggressively before any expensive computation.
- Log everything to `{ROOT}/logs/` as JSONL.
- Never run destructive operations without user confirmation.
- Fallback priority if compute is tight: preserve Alljoined Variant A (zero-shot transfer); drop DINOv2 ablation and Variants B/C first.
