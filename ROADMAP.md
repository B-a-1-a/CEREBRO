# Project Roadmap


## Phase 0 — Repo & Environment

- [ ] Create GitHub repo (private)
- [ ] Set up `.gitignore` (checkpoints, embeddings, `__pycache__`, `.ipynb_checkpoints`)
- [ ] Add `requirements.txt` (torch, transformers, mne, numpy, scipy, matplotlib, einops, timm)
- [ ] Set up shared Google Drive folder for checkpoints and cached embeddings
- [ ] Confirm all team members can run a Colab notebook that mounts the Drive

**Decision:** All large artifacts (checkpoints, `.npy` embedding caches) live in Google Drive, not in git.

---

## Phase 1 — Data Download & Sanity Check

**Goal:** Preprocessed THINGS-EEG2 data loaded into tensors, pipeline verified correct.

- [ ] Download preprocessed THINGS-EEG2 from OSF (Perceptogram repo, resource `anp5v`)
  - Sub-01 through Sub-10 as separate zips, loaded directly in Colab
  - Use the authors' preprocessed version (63ch, 250 Hz, 0–1000 ms epochs, MNN applied)
- [ ] Load one subject into a `(n_images, n_channels, n_timepoints)` tensor
- [ ] Average the 4 training repetitions per image → shape `(16540, 63, 250)` per subject
- [ ] Average the 80 test repetitions per image → shape `(200, 63, 250)` per subject (eval target)
- [ ] **Sanity check:** Reproduce the reported noise ceiling with a simple linear decoder on Subject 1
  - If your linear baseline Pearson R is in the ballpark of the THINGS-EEG2 paper, the pipeline is wired correctly
- [ ] Checkpoint: save per-subject tensors to Drive as `.npy` memmaps

**Expected time:** 1–2 days, 2 people.

---

## Phase 2 — Image Embedding Extraction

**Goal:** Cached CLIP and DINOv2 embeddings for all 16,740 unique images.

- [ ] Download the THINGS image stimuli (same OSF/OpenNeuro source as EEG data)
- [ ] Extract **CLIP ViT-L/14** embeddings for all 16,740 images (training + test)
  - Use `openai/clip-vit-large-patch14` via HuggingFace
  - Pool the penultimate layer or use the standard image embedding output
  - Save as numpy memmap: `clip_embeddings.npy`, shape `(16740, 768)`
- [ ] Extract **DINOv2** embeddings for the same images
  - Use `facebook/dinov2-large` via HuggingFace
  - Save as `dinov2_embeddings.npy`, shape `(16740, 1024)`
- [ ] Checkpoint both to Drive

**Strategy:** Run all CLIP experiments first to establish baseline. Swap to DINOv2 memmap once pipeline is stable.

**Expected time:** ~15–30 min on H100 / ~2–3 hours on T4. One person.

---

## Phase 3 — Three Models

### 3a — Per-Subject Linear Baseline

- [ ] For each of 10 subjects, train a linear projection: `CLIP_embedding (768,) → EEG_epoch (63×250,)`
- [ ] Loss: MSE. Evaluate by Pearson correlation on test set.
- [ ] Log per-subject R values. This is the floor.
- [ ] Checkpoint: save 10 model files + results dict to Drive

### 3b — Per-Subject Transformer

- [ ] Architecture: image embedding + RSVP context ring (±5 adjacent embeddings) → small transformer (4–6 layers, 4–8 heads, d_model 256–384) → linear head → EEG epoch
- [ ] Train independently for each of 10 subjects (same architecture, separate weights)
- [ ] Same eval as 3a
- [ ] This isolates: does the transformer help, independent of cross-subject sharing?
- [ ] Checkpoint: save 10 models + results

### 3c — Multi-Subject Transformer + Subject Block

- [ ] Same transformer trunk as 3b, but train jointly on all 10 subjects
- [ ] Subject block: `(10, d_model, 63×250)` tensor indexed by subject ID
  - Subject ID is passed as input; the block projects transformer output to that subject's channel space
- [ ] Train jointly, evaluate per-subject on test set
- [ ] This is the TRIBE-analog and the headline result
- [ ] Checkpoint: save model + per-subject results

**RSVP context window note:** For each trial, the input is the current image embedding + the embeddings of the 5 images before and 5 after in the RSVP stream. This accounts for neural contamination between adjacent RSVP stimuli — the key architectural departure from TRIBE's 100-second sliding window.

---

## Phase 4 — Evaluation & Figures

**Goal:** Reproduce TRIBE Figure 2D style comparison + neuroscientifically interpretable visualizations.

- [ ] **Box plot (main figure):** 3 columns × 10 dots (one per subject), linear / per-subject transformer / multi-subject transformer. Paired t-tests with FDR correction.
- [ ] **Channel × timepoint heatmap:** Pearson R plotted over the 63-channel × 250-timepoint space for the best model. Occipital channels should show highest R, especially in the 100–300 ms window (P1, N170).
- [ ] **Embedding ablation:** Swap CLIP → DINOv2, re-run multi-subject transformer, compare R. One additional row/column in the results table.
- [ ] Write up results section

---

## Phase 5 — Alljoined Transfer Experiment (bonus)

**This is the most novel part of the project.** No prior paper has cleanly demonstrated cross-hardware transfer within a single TRIBE-style model.

Three variants to try in order:

- [ ] **Variant A — Zero-shot on intersected channels:** Subset both THINGS-EEG2 and Alljoined to the ~20 channels present in both montages. Run the trained multi-subject transformer directly on Alljoined test subjects, no fine-tuning. Measure how much R drops vs. in-distribution subjects.
- [ ] **Variant B — Subject-block-only fine-tuning:** Freeze the transformer trunk. Add new subject block entries for Alljoined subjects. Fine-tune only the new subject block rows on a small subset of Alljoined trials. Compare to Variant A.
- [ ] **Variant C — Full fine-tuning with small LR:** Unfreeze the full model, fine-tune at ~10% of original LR on Alljoined. Compare to Variants A and B.

**Target result:** Even if R drops from THINGS-EEG2 levels, showing that Variant B or C stays above the per-subject linear baseline trained from scratch on Alljoined is the key claim.

---

## Phase 6 — Report & Presentation

### Report structure
- **Introduction:** TRIBE-fMRI-to-EEG transfer question; EEG as the deployable modality
- **Related work:** TRIBE, THINGS-EEG2, ENIGMA, Alljoined, Antonello fMRI scaling
- **Methods:** Dataset, architecture (transformer + subject block + RSVP context window), three-model comparison, embedding choices
- **Results:** Box plot, heatmap, embedding ablation, Alljoined transfer
- **Conclusions:** Subject-block trick generalizes to EEG; cross-hardware transfer implications for BCI

### Figures checklist
- [ ] Figure 1: Architecture diagram (transformer + subject block)
- [ ] Figure 2: Main comparison box plot (mirrors TRIBE Fig 2D)
- [ ] Figure 3: Channel × timepoint Pearson R heatmap
- [ ] Figure 4: Alljoined transfer results (Variants A/B/C vs. baselines)
- [ ] Table 1: Summary of all model × embedding combinations

---

## Compute Budget

| Task | Estimate | Where |
|---|---|---|
| Embedding extraction (CLIP + DINOv2, one-time) | ~30 min | H100 / Colab |
| Per training run (any of 3 models) | 4–8 hours | T4 / A10 |
| Full experiment suite (3 models × 3 seeds) | ~30–50 GPU-hours | Colab / Vast.ai |
| Alljoined variants (3 runs) | ~10–15 GPU-hours | H100 preferred |
| **Total** | **~50–70 GPU-hours** | |
| **Estimated cost if Colab runs out** | **< $50** | RunPod / Vast.ai spot |

---

## If Things Slip

Drop in this order:
1. DINOv2 ablation (nice to have, not essential)
2. Alljoined Variant C (full fine-tuning)
3. Alljoined Variant B (subject-block fine-tuning)
4. Alljoined Variant A (zero-shot) ← try hard to keep this

The **three-model comparison on THINGS-EEG2** is the non-negotiable core of the project.
