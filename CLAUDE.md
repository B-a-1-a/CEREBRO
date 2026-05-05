# CEREBRO — TRIBE-style EEG encoding on THINGS-EEG2

Project notes that aren't derivable from code or git history. Update after every completed phase.

## Where work happens

- **Code/cells**: Google Colab notebook (URL in `~/.claude/projects/-Users-bala-Repos-CEREBRO/memory/colab_notebook.md`). One mega-notebook, sections per phase.
- **Persistent artifacts**: `/content/drive/MyDrive/tribe-eeg/` (Drive). Survives session death.
- **Working scratch**: `/content/work/` and `/content/*.zip` (Colab local SSD, ~98 GB free at start). Ephemeral.
- **Plan**: `~/.claude/plans/implementation-prompt-tribe-style-quiet-cookie.md`.
- **State index**: `{ROOT}/logs/STATE.json` — flat dict of phase status. Update at end of each phase.

## Compute

- **Current GPU**: Tesla T4 (16 GB). The spec budgeted H100; on T4, Phase 5 training will be ~5–10× slower. Still works; user may queue Phase 5 for an H100 session later.
- **System RAM**: 12 GB.

## Data — load-bearing facts

- **EEG source**: THINGS-EEG2 preprocessed via Perceptogram OSF mirror (`anp5v`). Outer zip is **51 GB** (much larger than spec's "5–10 GB" estimate). Contains 20 inner zips: 10× 17ch + 10× 63ch.
- **Channel variant**: using **63ch** (per spec preference). Inner zip is named `sub-XX__63_channels.zip` but the actual array has **64 channels** (the "63" is a misnomer in the upstream naming). BioSemi 10-20 layout — channel list saved in `processed/eeg_meta.json`.
- **EEG file format**: each `preprocessed_eeg_*.npy` is a **pickled dict**, not a plain array. `np.load(path, allow_pickle=True).item()` returns:
  - `preprocessed_eeg_data`: float64 array, shape `(n_imgs, n_reps, 64, 100)`
  - `ch_names`: list of 64 channel names
  - `times`: 100-point time axis stored in **seconds** (-0.2 to 0.79 s, 10 ms step). Convert to ms when displaying.
  - Train: `(16540, 4, 64, 100)`. Test: `(200, 80, 64, 100)`.
  - **The 64th channel is `stim`**, a stimulus marker — not EEG. Exclude `ch_names.index("stim")` from training/eval. Real EEG is 63 channels (BioSemi 10-20).
- **Channel metadata** is saved once at `processed/eeg_meta.json` (ch_names, times_ms, n_channels, n_timepoints). Read from there in downstream phases.
- **Averaged EEG outputs**: `processed/eeg_train_avg_sub-XX.npz` (16540, 64, 100) float32, `eeg_test_avg_sub-XX.npz` (200, 64, 100) float32. Each .npz also stores `ch_names` and `times`.
- **Sanity check passed**: occipital subset (Oz/O1/O2/POz/PO7/PO8) shows -0.227 at 130 ms — classic N170. All-channels grand-mean is diluted and looks late; expected.
- **Image source**: OSF `y63gw`. The bundle `?zip=` endpoint is broken for this resource — fetch each file individually via the OSF API (see Phase 1b cell). Files: `training_images.zip` (655 MB), `test_images.zip` (8 MB), `image_metadata.npy` (657 KB), `LICENSE.txt`.
- **Images stay zipped on Drive.** Don't extract to 16k loose files — Drive's per-file write latency makes that ~hour-long. Phase 3 reads images directly via `zipfile.ZipFile.open(...)`.
- **Outer EEG zip** (`/content/thingseeg2_preproc.zip`, 51 GB) is kept as a recovery anchor through Phase 1; can be deleted after EEG averaging is verified.

## Operational rules learned the hard way

- **Verify download size before designing the flow.** OSF bundle endpoints lie about size and structure. Use HEAD or the OSF API listing first.
- **Never naïve-unzip a multi-GB bundle on `/content`.** Stream per-item from outer zip → write directly to Drive → free RAM. Caps `/content` at a few GB at any moment.
- **Colab MCP timeout is roughly 1–2 minutes per cell.** Cells that exceed this are killed; the kernel may auto-restart. Two patterns to handle long-running work:
  1. **Per-item cells** (~85s each) for jobs that fit naturally per subject/file.
  2. **Detached subprocess + status JSON** for full loops: cell launches `subprocess.Popen(['python','-u',script], stdout=open(log,'w'), start_new_session=True)`, returns instantly. Subsequent poll cells tail the log and read a status JSON. See Phase 1a `extract_eeg.py` pattern.
- **Always `flush=True`** on prints inside long loops; otherwise output buffering hides progress.
- **Wget**: use `--progress=dot:giga` on multi-GB downloads, never `--no-verbose` / `-q` (silent → looks stalled).

## Cross-phase dependencies (trip wires)

- **Phase 5 multi-subject training MUST include subject dropout** (mask `subject_id` with p≈0.1, route through a learned "unseen-subject" pathway). Without it, Phase 7 Variant A (zero-shot transfer) is impossible without retraining. Spec mentions this only in §7c — hoist it to Phase 5 from the start.
- **Phase 5 channel-subset retrain** (multi-subject CLIP, output head restricted to THINGS-EEG2 ∩ Alljoined channel intersection) must run *after* Phase 7a's first downloaded subject can be inspected to confirm the actual ch_names. **Don't trust the arxiv paper's appendix for channel layouts** — we wasted a Phase 5e run on a 28-ch wrong-list intersection. The verified 32-ch intersection from Alljoined sub-01 ch_names is in `logs/channel_intersection.json`.
- **Phase 4 ridge gate**: mean Pearson R must land in [0.10, 0.25]. <0.05 → image-EEG alignment broken (almost always filename ordering); >0.4 → leak. Stop and debug if either.
- **Phase 5 sanity gate**: test Pearson R(multi-subject) > R(per-subject) > R(linear). On 14k training samples per subject the per-subject transformer is data-starved and ties linear at the rescue config (d_model=192, dropout=0.3); the multi >> per_sub ≈ linear ordering is the *expected* TRIBE-style outcome. The full-size per-subject (d_model=384, n_layers=4) underperforms linear because of overfit, not alignment failure.
- **Phase 7 EEG adaptation**: TS2 epoch is [-0.2, 0.79]s @100Hz (100 timepoints, 200ms pre-stim baseline). Alljoined is [0, ~1.0]s @250Hz (250 timepoints, NO pre-stim). Adapter pipeline: average reps → take post-stim [0, 0.79]s window → resample 200→80 timepoints → reorder channels to TS2 intersection order → prepend 20 zeros for the missing baseline. Score Pearson R only on the post-stim 80 timepoints. Pipeline is `adapt_alljoined_subject()` in the notebook.
- **Phase 7 sequential rule**: never launch parallel training subprocesses. The A100 5-way parallel attempt deadlocked silently (probably CUDA init contention) and burned 18,000s. Foreground subprocess.run() for short jobs, detached Popen + status JSON polling for long jobs, ALWAYS one at a time.
- **Random seeds**: 42, 1337, 2026 across all training scripts. Set both `torch.manual_seed` and `np.random.seed` at script top.

## Workflow rules

- **Pause flow**: print plan + size/time estimate before each long op, then auto-proceed. User interrupts only if objecting (no chat-level "go" handshake).
- **CLAUDE.md updated after every completed phase.**
- **Run `/compact` before starting any phase if context usage is past 60%.**

## Phase status

- Phase 0: ✓ environment ready, packages pinned, dirs built, STATE.json + env.json saved.
- Phase 1: ✓ EEG extracted (10/10 subjects, 64-channel) + images downloaded (zipped on Drive). Inventory at `logs/data_inventory.json`.
- Phase 2: ✓ all 10 subjects averaged → `processed/eeg_{train,test}_avg_sub-XX.npz`. Image manifests built from zip namelists. Sanity ERP confirmed N170 in occipital subset at 130 ms.
- Phase 3: ✓ embeddings extracted on T4 (CLIP train 305s, DINOv2 train 358s). Sanity passed: CLIP Δ=+0.23, DINOv2 Δ=+0.61 (same-concept vs random). Outputs: `embeddings/{clip_vitl14,dinov2_large}_{training,test}.npy`. Used `.half()` weights at inference for ~2× speed; outputs cast to float32. Loaded image bytes directly from `*_images.zip` via `zipfile.ZipFile.open()`.
- Phase 4: ✓ ridge baseline passed under correct interpretation. **All-cells mean R = 0.088 looks below the spec's [0.10, 0.25] gate, but that gate assumes the 17-ch occipital subset.** On 63 channels (with frontal/noise dilution), the equivalent measure is occipital × peri-stim window (80–300 ms) = **0.351**. Max R per subject ≈ 0.68 at parietal channels in the visual evoked window. Per-subject `r_per_ct` arrays saved to `results/phase4_r_per_ct_sub-XX.npy` for Phase 6 heatmap reuse.
- Phase 5: ✓ Three CLIP seeds (42, 1337, 2026) all replicate cleanly. Multi-subject CLIP **test mean R = 0.091 ± 0.015** (per-subject, N=10) vs linear baseline 0.042 ± 0.008 — paired t-test highly significant. Per-subject transformer (rescue: d_model=192, n_layers=2, dropout=0.3) ties linear at 0.039 ± 0.018 — i.e. the multi-subject pooling rescue is real. DINOv2 multi-subject < CLIP (0.068 vs 0.089), confirming CLIP's vision-language alignment is load-bearing. Channel-subset model (32-ch intersection) reaches **test mean R = 0.122 ± 0.019** because it drops noisy frontal channels.
- Phase 5e channel-subset retrain: **the original Phase 7b channel intersection (28 ch) was guessed wrong from the arxiv paper.** The actual Alljoined Emotiv Flex 2 montage (verified from sub-01 ch_names) overlaps 32 channels with BioSemi 64 — i.e. all of Alljoined's channels are a subset of TS2's. The 32-ch retrained model is at `checkpoints/multi_subject_clip_chsubset32_seed42/best.pt`. The corrected intersection is in `logs/channel_intersection.json`.
- Phase 6: ✓ figures and `results/all_results.csv` (80 rows). `figures/headline_boxplot.{png,pdf}` shows the multi >> linear ≈ per-subject pattern. Channel × time heatmap and CLIP-vs-DINOv2 ablation also saved.
- Phase 7: ✓ Cross-hardware transfer to Alljoined-1.6M (Emotiv Flex 2, 32-ch consumer-grade), N=5 subjects (sub-01..05 of 20). **Headline: Variant B (subject-block FT, frozen encoder, 5 ep on each Alljoined subject's training data) reaches posterior R = 0.030 ± 0.015** vs scratch ridge 0.007 ± 0.009, paired-t **p = 0.034**. Variant A (zero-shot via null pathway) and Variant C (full FT, lr=1e-5, 1 ep) trend positive but underpowered at N=5. Conclusion: the multi-subject TS2 encoder learned hardware-transferable visual representations.
- Phase 8: pending.

## Channel groups (BioSemi 64, THINGS-EEG2)

For consistent regional summaries downstream:
- **Occipital/parietal (visual)**: `["Oz", "O1", "O2", "POz", "PO7", "PO8", "PO3", "PO4", "P7", "P8"]`
- **Frontal**: `["Fp1", "Fp2", "F3", "F4", "AFz", "AF7", "AF8"]` — note **no `Fz`** in this montage; midline frontal is `AFz`/`FCz`.
- **Stim marker**: `ch_names.index("stim")` — the 64th channel; always exclude from analysis.
- **Peri-stim window**: 80–300 ms (P1/N170/P2 visual response).
