# TRIBE-EEG: Foundation Model Encoding for Event-Related EEG

**Course:** Intro to Foundation Models — Final Project  
**Team:** Longjin Che, Ian Gloege, Bala Shukla, Rishav Roy

---

## Overview

This project investigates whether the core architectural insight of Meta's **TRIBE v2** — that a subject-conditioned transformer trained jointly across many participants outperforms per-subject models, and scales log-linearly with data — transfers from slow fMRI signals to fast, event-related EEG.

We use the **THINGS-EEG2** dataset (10 subjects, ~820k trials, 63 channels) as our primary testbed, with RSVP visual stimuli encoded via **CLIP ViT-L/14** (primary) and **DINOv2** (ablation). A bonus experiment tests zero-shot transfer to the **Alljoined-1.6M** dataset, recorded on consumer-grade Emotiv hardware — the most novel contribution of the project.

**One-sentence claim:** A subject-conditioned transformer trained jointly on THINGS-EEG2 will predict held-out EEG responses to natural images more accurately than either a per-subject linear baseline or a per-subject transformer, and a learned subject block can absorb cross-hardware heterogeneity when transferring to Alljoined.

---

## Repository Structure

```
tribe-eeg/
├── README.md
├── ROADMAP.md
├── .gitignore
│
├── notebooks/               # Colab-ready notebooks (primary interface)
│   ├── 01_data_download.ipynb
│   ├── 02_embedding_extraction.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_transformer_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_alljoined_transfer.ipynb
│
├── src/                     # Core Python modules (imported by notebooks)
│   ├── data/
│   │   ├── things_eeg2.py       # Dataset class for THINGS-EEG2
│   │   └── alljoined.py         # Dataset class for Alljoined
│   ├── models/
│   │   ├── baselines.py         # Per-subject linear baseline
│   │   ├── transformer.py       # TRIBE-analog transformer + subject block
│   │   └── subject_block.py     # Subject-conditional projection layer
│   ├── training/
│   │   ├── trainer.py           # Training loop with checkpointing
│   │   └── losses.py            # MSE loss + Pearson correlation metric
│   └── eval/
│       └── metrics.py           # Evaluation utilities, figure generation
│
├── configs/                 # Hyperparameter configs (YAML)
│   ├── baseline.yaml
│   ├── transformer_per_subject.yaml
│   └── transformer_multi_subject.yaml
│
├── scripts/                 # Standalone scripts for batch jobs
│   ├── extract_embeddings.py
│   └── run_experiment.py
│
└── checkpoints/             # Git-ignored; lives in GDrive/GCS
```

---

## Datasets

| Dataset | Subjects | Trials | Channels | Hardware | Role |
|---|---|---|---|---|---|
| THINGS-EEG2 | 10 | ~820k | 63 | BioSemi (research-grade) | Primary training/eval |
| Alljoined-1.6M | 20 | ~1.6M | 32 | Emotiv Flex 2 (consumer) | Transfer experiment |

All data is loaded remotely (Google Drive / OSF / OpenNeuro) — nothing downloads to local machine.

---

## Models

Three models are trained and compared, mirroring TRIBE's Figure 2D:

1. **Per-subject linear baseline** — CLIP embedding → linear projection → EEG epoch, trained independently per subject. Floor.
2. **Per-subject transformer** — Same architecture as (3) but trained independently per subject. Isolates "does the transformer help" from "does cross-subject training help."
3. **Multi-subject transformer + subject block** *(TRIBE-analog)* — Shared transformer trunk with a learned subject-conditional output projection. This is the headline model.

---

## Key Design Choices

- **RSVP context window:** TRIBE uses 100-second sliding windows for slow fMRI. We replace this with a ring of ±5 RSVP-adjacent image embeddings, addressing the temporal contamination specific to fast RSVP paradigms.
- **Subject block:** A `(N_subjects, d_model, n_channels × n_timepoints)` tensor indexed by subject ID. Same design as TRIBE; extended to absorb hardware differences in the Alljoined transfer.
- **Embeddings:** CLIP ViT-L/14 as primary (validated on THINGS-EEG2 by prior work); DINOv2 as ablation (self-supervised, closer to TRIBE's V-JEPA-2 philosophy).

---

## Compute

All experiments fit within free Colab tier or a modest RunPod/Vast.ai budget:

- Embedding extraction: ~15 min on H100 (one-time)
- Per training run: ~4–8 hours on T4/A10
- Full experiment suite: under 50 GPU-hours total
- Estimated cost if Colab runs out: **< $50**

---

## Setup

```bash
git clone https://github.com/YOUR_ORG/tribe-eeg.git
cd tribe-eeg
pip install -r requirements.txt
```

All data loading happens in-notebook via remote URLs. See `notebooks/01_data_download.ipynb`.

---

## References

- Toneva et al., TRIBE v2 (Meta FAIR, 2024)
- Gifford et al., THINGS-EEG2 (2022)
- Xu et al., Alljoined-1.6M (2025)
- Conwell et al., ENIGMA (2023)
