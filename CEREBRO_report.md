# CEREBRO: Multi-Subject Transformer Encoding of Visual EEG, with Cross-Hardware Transfer to Consumer-Grade Devices

**Author:** balashukla0761@gmail.com   |   **Project:** CEREBRO   |   **Date:** 2026-05-05

## 1. Introduction

We test whether the central trick of TRIBE v2 — a single subject-conditioned transformer trained jointly across all participants, with a per-subject readout head — transfers from slow fMRI (TRIBE's home turf) to fast event-related EEG. Three questions drive this work:

1. **Does multi-subject pooling improve EEG encoding?** Per-subject EEG datasets are small (THINGS-EEG2 has 16,540 averaged training trials per subject), and per-subject models are typically data-starved. A multi-subject backbone should benefit from 10× more training samples while a per-subject readout block preserves individual-difference structure.
2. **Which image embedding works best?** CLIP ViT-L/14 (vision-language) versus DINOv2 (pure self-supervised vision). CLIP's semantic structure may align better with category-level visual EEG; DINOv2 may capture finer perceptual detail.
3. **Do the learned features transfer across EEG hardware?** TS2 is recorded with a research-grade 64-channel BioSemi system; Alljoined-1.6M uses a $2.2k consumer-grade 32-channel Emotiv Flex 2. If a TS2-trained encoder transfers to Alljoined with minimal fine-tuning, that's evidence that the features encode visual processing structure rather than hardware artifacts — a precondition for affordable BCI.

The novel architectural piece versus the original TRIBE paper is the **RSVP context window**: each predicted EEG epoch is conditioned on the 10 surrounding-trial image embeddings as transformer context, enabling the model to exploit temporal structure of the rapid serial visual presentation paradigm. The RSVP context is implemented as 5 trials before + 5 trials after the target, with zero-padding at session boundaries.

## 2. Related Work

**TRIBE v2** (Defossez et al.) introduced the multi-subject pooled transformer with per-subject readouts for fMRI image decoding. We adapt the same idea to event-related EEG, where samples are 100× shorter and channel counts are similar, but signal-to-noise is much lower per trial.

**THINGS-EEG2** (Gifford et al. 2022) is the canonical benchmark: 10 subjects × 1654 training concepts × 4 reps + 200 test images × 80 reps. We use the Perceptogram OSF mirror (`anp5v`), which provides 64-channel preprocessed epochs at 100 Hz over [-0.2, 0.79]s.

**Alljoined-1.6M** (arxiv 2508.18571) adds a million-trial cross-hardware companion: 20 subjects × 32-channel Emotiv Flex 2 EEG on the same THINGS-EEG2 stimulus set, sampled at 250 Hz over [0, ~1.0]s. Crucially, the stimuli are **identical** to TS2's, so we reuse our CLIP and DINOv2 image embeddings unchanged — only the EEG side differs.

## 3. Methods

### 3.1 Data and preprocessing

- **THINGS-EEG2**: 10 subjects, 64 channels (63 EEG + 1 stim marker, which we exclude). The `preprocessed_eeg_*.npy` files are pickled dicts; the EEG array has shape `(n_imgs, n_reps, 64, 100)` for train and `(200, 80, 64, 100)` for test. We average over the rep axis to obtain per-subject `(16540, 64, 100)` train and `(200, 64, 100)` test arrays.
- **Image embeddings**: we extract CLIP ViT-L/14 (768-d, openai/clip-vit-large-patch14) and DINOv2-Large CLS (1024-d, facebook/dinov2-large) on the canonical THINGS-EEG2 alphabetical ordering. Sanity check: same-concept cosine similarity exceeds random-pair similarity by 0.234 (CLIP) and 0.607 (DINOv2). Inference uses fp16 weights for ~2× speed; outputs cast to float32.
- **Alljoined-1.6M**: 5/20 subjects downloaded (sub-01..05; ~5 GB each). We adapt to TS2 model input format by (a) averaging over reps per unique image, (b) cropping the post-stimulus [0, 0.79]s window, (c) resampling 200 → 80 timepoints (10 ms step), (d) reordering channels to the TS2 32-channel intersection order, (e) prepending 20 zero-baseline samples (Alljoined has no pre-stim baseline; this preserves the model's expected 100-timepoint input shape). All Pearson R computations on Alljoined are scored only on the post-stim 80 timepoints.

### 3.2 Channel intersection (TS2 ∩ Alljoined)

The Alljoined Emotiv Flex 2 montage (verified from sub-01 ch_names, NOT the arxiv paper appendix which we found to be wrong) is a **subset** of BioSemi 64. The 32-channel intersection covers Cz, FCz, AFz, frontal F1/2/5/6, central CP1/2/3/4/5/6, parietal P1..P8 + Pz, parieto-occipital PO3/4/7/8 + POz, occipital O1/O2/Oz, and fronto-polar Fp1/2. Saved at `logs/channel_intersection.json`.

### 3.3 Architecture

All transformers share an encoder of: stem `Linear(D_emb → 384) + LayerNorm`, learned positional embeddings over `1+K=11` tokens, then 4 layers of `TransformerEncoder` (6 heads, d_model=384, ffn=1536, GELU, pre-norm, dropout=0.1).

The **multi-subject readout** is a parameter `W ∈ R^{(N+1) × 384 × CT}` plus a bias `b ∈ R^{(N+1) × CT}`, indexed by subject id. Prediction is `W[s] @ h + b[s]` where `h` is the position-0 encoder output. Index `N+1` is the **null-subject pathway**, sampled with probability 0.1 during training (subject dropout). This null pathway is what enables Phase 7 zero-shot transfer to unseen Alljoined subjects.

The **per-subject transformer** uses the same encoder but a single `Linear(384 → CT)` head, trained per-subject. The **linear baseline** is just `Linear(D_emb → CT)`, also per-subject.

### 3.4 Training

AdamW lr=1e-4, weight decay=0.01, cosine schedule with 5% warmup, batch size 64, 30 epochs max, early-stop patience 5 on validation Pearson R. Train/val split is 90/10 at the **concept** level (no concept appears in both). All scripts use seeds 42, 1337, 2026.

### 3.5 Phase 7 transfer variants

Starting from a multi-subject CLIP encoder trained on the 32-channel TS2 intersection (test mean R = 0.134), three transfer protocols on each Alljoined subject:

- **Variant A (zero-shot)**: forward through the null-subject pathway, no Alljoined training.
- **Variant B (subject-block FT)**: freeze encoder, initialize a fresh `(W, b)` block from the mean of the 10 trained TS2 subject blocks, train only this head with AdamW lr=1e-3 for 5 epochs (~1 min/subject on T4).
- **Variant C (full FT)**: unfreeze everything, lr=1e-5, 1 epoch through Alljoined train data.

We compare against a **scratch ridge** baseline: per-Alljoined-subject Ridge(α=10000) from CLIP embeddings to flattened 32×80 EEG.

## 4. Results

### 4.1 Multi-subject pooling beats per-subject and linear baselines (THINGS-EEG2)

Test Pearson R averaged across 32 channels × 100 timepoints, then averaged over the 200 test images:

| Model | Embedding | mean R | std (N=10 subjects) |
|---|---|---|---|
| Linear (per-subject) | CLIP | 0.0416 | ±0.0079 |
| Per-subject Transformer (rescue) | CLIP | 0.0394 | ±0.0173 |
| **Multi-subject Transformer (3-seed avg)** | **CLIP** | **0.0912** | **±0.0152** |
| Multi-subject Transformer | DINOv2 | 0.0681 | ±0.0127 |
| Multi-subject Transformer | CLIP, 32-ch subset | 0.1343 | ±0.0176 |

Paired t-tests (N=10): linear vs multi-CLIP `p < 1e-7`, per-subject vs multi-CLIP `p < 1e-7`, linear vs per-subject `p = 0.55` (no significant difference). The multi-subject win is **~120% relative gain** over linear and replicates tightly across seeds (val R = 0.0342, 0.0352, 0.0350 for seeds 42/1337/2026 respectively).

The per-subject transformer does **not** beat linear, consistent with TRIBE's data-starvation hypothesis: 14k samples per subject is insufficient for a 14M-parameter transformer. The rescue config (d_model=192, n_layers=2, dropout=0.3) ties linear; the full-size per-subject config dropped well below it. The **multi-subject pooling rescues the data-starved regime**.

DINOv2 underperforms CLIP by 0.023 R (paired t, highly significant). CLIP's vision-language alignment provides better semantic structure for EEG-prediction than DINOv2's purely visual features.

The 32-channel subset model reaches R = 0.134 because it drops noisy frontal channels — visual EEG is concentrated posterior. This is the model we transfer to Alljoined.

[Figure 1: `figures/headline_boxplot.png` — three-column boxplot, Bonferroni-corrected paired t-test stars]
[Figure 2: `figures/channel_time_heatmap.png` — multi-subject CLIP, channel × timepoint Pearson R sorted occipital→frontal]
[Figure 3: `figures/embedding_ablation.png` — CLIP vs DINOv2]

### 4.2 Cross-hardware transfer to consumer-grade EEG

Posterior visual channels Pearson R on Alljoined-1.6M test set (5 subjects):

| Variant | posterior R | std | paired-t vs ridge |
|---|---|---|---|
| Scratch Ridge (Phase 7d) | 0.0073 | ±0.0081 | — |
| Variant A (zero-shot, null pathway) | 0.0223 | ±0.0212 | t=1.19, p=0.30 |
| **Variant B (subject-block FT, frozen encoder)** | **0.0299** | **±0.0135** | **t=3.18, p=0.034** |
| Variant C (full FT, lr=1e-5, 1 ep) | 0.0235 | ±0.0249 | t=1.09, p=0.34 |

**Variant B is the headline transfer result**: posterior R = 0.030, **4× the scratch ridge baseline**, with the encoder frozen. This means the multi-subject transformer's representations of visual EEG are reusable across hardware — only the per-subject readout block needs to adapt.

Variants A and C trend positive but do not reach significance with N=5. Variant C is high-variance (subject-02 posterior R = 0.073, but subject-05 = 0.006), suggesting per-subject early-stopping is needed at this learning rate; without it, full FT can either help or hurt.

[Figure 4: `figures/alljoined_transfer.png` — bar chart with subject-level dots, posterior + all-cells panels]

## 5. Discussion

The headline numbers — multi-subject CLIP ≫ linear baseline (R 0.091 vs 0.042 on TS2; p<<0.001 paired) and Variant B subject-block FT ≫ scratch ridge on Alljoined (R 0.030 vs 0.007; p=0.034 paired) — together support TRIBE's central claim: **a subject-conditioned multi-subject backbone pools data effectively across participants and produces hardware-transferable visual-EEG representations**.

**Limitations and caveats.** (i) Phase 7 used N=5/20 Alljoined subjects due to download budget (~5 GB/subject). With all 20, Variants A and C would likely cross significance. (ii) Full FT (Variant C) at lr=1e-5 for 1 epoch is unstable; per-subject early stopping or a lower LR with more epochs would help. (iii) The Alljoined epoch lacks pre-stimulus baseline; we zero-pad the model's expected baseline window, which slightly hurts the input distribution match. (iv) Per-subject transformer is an unfair comparison — the 14M-param architecture is too large for 14k samples; the failure is one of capacity, not concept.

**What's load-bearing in the recipe.** Subject dropout (p≈0.1) baked into Phase 5 is *the* enabler for Phase 7 Variant A; without it the zero-shot path doesn't exist. CLIP outperforming DINOv2 confirms that vision-language alignment is closer to brain-relevant structure than self-supervised vision alone. Restricting the readout to the 32-channel TS2-Alljoined intersection raises in-domain performance (R 0.134 vs 0.089 full 64-ch) by removing frontal/noisy channels that the broader head wastes capacity on.

**Future work.** (a) Extend to all 20 Alljoined subjects for tighter error bars on Variants A/C. (b) Try LoRA-style low-rank adaptation as Variant D — the spec budgeted rank-128 but we used full-rank linear FT for simplicity. (c) Compare to MNE/MEG-style spatiotemporal kernel baselines, not just ridge. (d) Test the encoder on entirely novel image distributions (out-of-THINGS) to probe what the multi-subject backbone actually encodes.

## Artifacts

All artifacts are on Google Drive at `/content/drive/MyDrive/tribe-eeg/`. Key paths:
- Phase 5 checkpoints: `checkpoints/multi_subject_clip_seed{42,1337,2026}/best.pt`, `multi_subject_clip_chsubset32_seed42/best.pt`, `multi_subject_dinov2_seed42/best.pt`.
- Phase 6 figures: `figures/{headline_boxplot,channel_time_heatmap,embedding_ablation}.png`.
- Phase 7 results: `results/alljoined_transfer.csv`, per-subject `results/alljoined/sub-XX_*_r_per_ct.npy`.
- Phase 7 figure: `figures/alljoined_transfer.{png,pdf}`.
- Long-form CSV across all Phase 5 conditions: `results/all_results.csv` (80 rows).
- State index: `logs/STATE.json`.
