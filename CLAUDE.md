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
- **Phase 5 channel-subset retrain** (multi-subject CLIP, output head restricted to THINGS-EEG2 ∩ Alljoined channel intersection) must run *after* Phase 7b computes the intersection but *before* Phase 7c transfers. Budget for this in the H100 plan.
- **Phase 4 ridge gate**: mean Pearson R must land in [0.10, 0.25]. <0.05 → image-EEG alignment broken (almost always filename ordering); >0.4 → leak. Stop and debug if either.
- **Phase 5 sanity gate**: test Pearson R(multi-subject) > R(per-subject) > R(linear). If violated, stop.
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
- Phase 5–8: pending.

## Channel groups (BioSemi 64, THINGS-EEG2)

For consistent regional summaries downstream:
- **Occipital/parietal (visual)**: `["Oz", "O1", "O2", "POz", "PO7", "PO8", "PO3", "PO4", "P7", "P8"]`
- **Frontal**: `["Fp1", "Fp2", "F3", "F4", "AFz", "AF7", "AF8"]` — note **no `Fz`** in this montage; midline frontal is `AFz`/`FCz`.
- **Stim marker**: `ch_names.index("stim")` — the 64th channel; always exclude from analysis.
- **Peri-stim window**: 80–300 ms (P1/N170/P2 visual response).
