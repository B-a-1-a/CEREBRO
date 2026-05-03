# Implementation Prompt: TRIBE-Style EEG Encoding on THINGS-EEG2

You are Claude Code, connected via MCP to a Google Colab notebook. We are implementing a research project that tests whether TRIBE v2's central architectural insight — a subject-conditioned transformer trained jointly across all participants — transfers from slow fMRI to fast event-related EEG. This prompt is your full project spec. Read it once end-to-end before starting Phase 1.

## Core principles (read first, follow throughout)

**Nothing lives on the user's laptop.** All data downloads, model checkpoints, and intermediate artifacts live on Google Drive (persistent across Colab sessions) or Colab's `/content` local SSD (fast working scratch). The user only ever sees figures, logs, and the final report.

**Checkpoint aggressively.** Colab sessions die. Every phase below has explicit checkpoint files written to Drive. Before doing any expensive computation, *check whether the checkpoint already exists* and skip if so. Use a `--force` flag pattern: each phase script accepts a `force=False` parameter that, when True, recomputes from scratch.

**Drive is slow, /content is fast.** The pattern is: one-time download to Drive → at start of each session, copy/symlink relevant pieces to `/content/work/` → operate there → save artifacts back to Drive. Never run training loops directly off Drive-mounted paths.

**Sanity-check before scaling.** Each phase has a stated success criterion. If the criterion isn't met, stop and surface the issue rather than barreling into the next phase.

**Ask before destructive operations.** If you're about to delete or overwrite a checkpoint, confirm with the user.

---

## Directory layout (create this in Phase 0)

```
/content/drive/MyDrive/tribe-eeg/          # Persistent across sessions
├── raw/
│   ├── thingseeg2_preproc/                # sub-01/ ... sub-10/ from OSF
│   ├── thingseeg2_metadata/               # training_images/, test_images/, concepts CSVs
│   └── alljoined/                         # (filled in Phase 7)
├── processed/
│   ├── eeg_train_avg_sub-{01..10}.npz     # per-subject averaged training EEG
│   ├── eeg_test_avg_sub-{01..10}.npz      # per-subject averaged test EEG
│   ├── image_filenames_train.json         # ordered list, index → filename
│   └── image_filenames_test.json
├── embeddings/
│   ├── clip_vitl14_train.npy              # memmap (16540, 768)
│   ├── clip_vitl14_test.npy               # memmap (200, 768)
│   ├── dinov2_large_train.npy             # memmap (16540, 1024)
│   └── dinov2_large_test.npy
├── checkpoints/
│   ├── linear_baseline/sub-{01..10}.pt
│   ├── per_subject_transformer/sub-{01..10}.pt
│   ├── multi_subject_transformer/best.pt
│   └── alljoined_transfer/{zero_shot,subject_block_ft,full_ft}.pt
├── logs/                                  # JSONL training logs per run
├── results/                               # final eval JSON/CSV
└── figures/                               # plots for the report

/content/work/                              # Ephemeral fast scratch (re-create each session)
```

---

## Phase 0 — Environment setup

Goal: Colab session is mounted, GPU is verified, packages are installed, directory tree exists.

```python
# 1. Verify GPU
!nvidia-smi
# Expected: H100 if user provisioned it; otherwise T4/A100 — proceed regardless but warn if no GPU.

# 2. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Install packages (pin versions to avoid surprises)
!pip install -q torch torchvision transformers==4.44.0 \
    mne==1.7.1 numpy scipy scikit-learn pandas matplotlib seaborn \
    tqdm einops h5py datalad-installer

# 4. Create directory tree
import os
ROOT = "/content/drive/MyDrive/tribe-eeg"
WORK = "/content/work"
SUBDIRS = ["raw/thingseeg2_preproc", "raw/thingseeg2_metadata", "raw/alljoined",
           "processed", "embeddings",
           "checkpoints/linear_baseline", "checkpoints/per_subject_transformer",
           "checkpoints/multi_subject_transformer", "checkpoints/alljoined_transfer",
           "logs", "results", "figures"]
for d in SUBDIRS:
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
os.makedirs(WORK, exist_ok=True)
```

**Checkpoint:** print a confirmation that all directories exist and GPU is detected. Save GPU info to `{ROOT}/logs/env.json` for the report.

---

## Phase 1 — Data acquisition

Goal: preprocessed THINGS-EEG2 EEG and the THINGS image set are on Drive, never to be re-downloaded.

### 1a. Preprocessed EEG (from Perceptogram's OSF mirror — fastest path)

The preprocessed THINGS-EEG2 EEG is available as a single OSF zip. Each subject is a separate inner zip.

```python
import os, subprocess

EEG_OSF_URL = "https://files.de-1.osf.io/v1/resources/anp5v/providers/osfstorage/?zip="
EEG_DEST = f"{ROOT}/raw/thingseeg2_preproc"

# Skip if already present
expected_subjects = [f"sub-{i:02d}" for i in range(1, 11)]
already_have = all(os.path.exists(f"{EEG_DEST}/{s}") for s in expected_subjects)

if not already_have:
    # Download to /content (fast disk), then unzip to Drive
    !wget -O /content/thingseeg2_preproc.zip "$EEG_OSF_URL"
    !unzip -q /content/thingseeg2_preproc.zip -d /content/thingseeg2_preproc
    # Inner zips per subject
    for i in range(1, 11):
        sub = f"sub-{i:02d}"
        inner_zip = f"/content/thingseeg2_preproc/{sub}.zip"
        if os.path.exists(inner_zip):
            !unzip -q "$inner_zip" -d "$EEG_DEST/"
    !rm -rf /content/thingseeg2_preproc /content/thingseeg2_preproc.zip
else:
    print("EEG already downloaded, skipping.")
```

### 1b. Image metadata (from OSF)

```python
IMG_OSF_URL = "https://files.de-1.osf.io/v1/resources/y63gw/providers/osfstorage/?zip="
IMG_DEST = f"{ROOT}/raw/thingseeg2_metadata"

if not os.path.exists(f"{IMG_DEST}/training_images"):
    !wget -O /content/thingseeg2_metadata.zip "$IMG_OSF_URL"
    !unzip -q /content/thingseeg2_metadata.zip -d "$IMG_DEST"
    # The metadata zip itself contains training_images.zip and test_images.zip
    !cd "$IMG_DEST" && unzip -q training_images.zip && unzip -q test_images.zip && \
        rm training_images.zip test_images.zip
    !rm /content/thingseeg2_metadata.zip
```

### 1c. Sanity checks

After download, verify:
- 10 subject folders exist under `raw/thingseeg2_preproc/`
- Each subject folder contains `.npy` files (preprocessed EEG epochs) and `.json` label files
- `raw/thingseeg2_metadata/training_images/` contains 1654 concept folders, each with 10 images
- `raw/thingseeg2_metadata/test_images/` contains 200 concept folders, each with 1 image

Print a summary table of counts. Save to `{ROOT}/logs/data_inventory.json`.

**Checkpoint:** the EEG and images are on Drive. Subsequent sessions can skip Phase 1 entirely.

---

## Phase 2 — EEG preprocessing (averaging repetitions)

Goal: collapse the 4× training repetitions and 80× test repetitions into one EEG response per `(subject, image)` pair, save as compact `.npz` files.

### Expected input shapes (per subject, from THINGS-EEG2)

The preprocessed files from the authors give epoched data at 100 Hz (downsampled from 1000 Hz) with 100 timepoints spanning ~−200 to 800 ms. The exact format you should expect:
- Training: array shape `(16540, 4, 17, 100)` or similar — `(n_images, n_repetitions, n_channels, n_timepoints)`
- Test: array shape `(200, 80, 17, 100)`
- Channels: 17 channels selected by the authors (occipital/parietal subset). If you find the full 63-channel version in the data, use that — note which version you're using in the log.

> **Note**: You will need to inspect the actual file structure once it's downloaded — the authors' preprocessed format may differ slightly. Open one subject's files first, print shapes, and confirm before writing the averaging loop. If shapes don't match what's described above, surface this to the user before proceeding.

### Averaging logic

```python
import numpy as np, json
from tqdm import tqdm

PROC = f"{ROOT}/processed"

def average_subject(sub_id):
    """Load one subject, average reps, save .npz to Drive."""
    out_train = f"{PROC}/eeg_train_avg_{sub_id}.npz"
    out_test = f"{PROC}/eeg_test_avg_{sub_id}.npz"
    if os.path.exists(out_train) and os.path.exists(out_test):
        return  # already done

    sub_dir = f"{ROOT}/raw/thingseeg2_preproc/{sub_id}"
    # Load preprocessed arrays — adjust filenames after you inspect actual layout
    # Pseudocode:
    train = np.load(f"{sub_dir}/preprocessed_eeg_training.npy")  # (16540, 4, C, T)
    test = np.load(f"{sub_dir}/preprocessed_eeg_test.npy")        # (200, 80, C, T)

    train_avg = train.mean(axis=1).astype(np.float32)  # (16540, C, T)
    test_avg = test.mean(axis=1).astype(np.float32)    # (200, C, T)

    np.savez_compressed(out_train, eeg=train_avg)
    np.savez_compressed(out_test, eeg=test_avg)

for i in range(1, 11):
    average_subject(f"sub-{i:02d}")
```

### Image filename ordering

The image-EEG correspondence is implicit in the array index. You need an explicit mapping:

```python
# Walk training_images/ in the canonical order used by THINGS-EEG2
# (alphabetical by concept folder, then alphabetical within)
import os, json
train_imgs = []
for concept in sorted(os.listdir(f"{ROOT}/raw/thingseeg2_metadata/training_images")):
    cdir = f"{ROOT}/raw/thingseeg2_metadata/training_images/{concept}"
    for img in sorted(os.listdir(cdir)):
        train_imgs.append(f"{concept}/{img}")
# Should be exactly 16540
assert len(train_imgs) == 16540, f"Got {len(train_imgs)}"
with open(f"{PROC}/image_filenames_train.json", "w") as f:
    json.dump(train_imgs, f)
# Same for test (200 images)
```

### Sanity checks

- Per-subject averaged training tensor has shape `(16540, C, T)` with no NaNs
- Test tensor has shape `(200, C, T)`
- Image filename list has lengths 16540 and 200 respectively
- Quick spot-check: load `sub-01` test EEG, average across all 200 images and all channels, plot mean response. You should see a clear visual evoked response peaking around 100–200 ms post-stimulus. Save the plot to `figures/sanity_evoked.png`.

**Checkpoint:** `processed/` directory contains 20 `.npz` files plus 2 JSON manifests. Total size ~5–10 GB.

---

## Phase 3 — Image embeddings (CLIP primary, DINOv2 ablation)

Goal: every unique image (16,740 total) has a cached embedding from both CLIP ViT-L/14 (primary) and DINOv2-Large (ablation). Stored as numpy memmaps for fast indexed access during training.

### Why two encoders

- **CLIP ViT-L/14** is the primary. It has been validated repeatedly on THINGS-EEG2 in prior work, so reviewers will accept it as a no-questions-asked baseline. Build the entire pipeline against CLIP first.
- **DINOv2-Large** is the ablation. It's self-supervised and therefore a closer spiritual analog to TRIBE's V-JEPA-2 video encoder. Once the pipeline is stable on CLIP, swap in DINOv2 by changing one line in the dataloader and rerun the multi-subject transformer to see if self-supervised features yield higher Pearson correlation.

With H100 access, both encoders together take ~20 minutes. Extract both up front.

### Implementation

```python
import torch, numpy as np, json
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor
from tqdm import tqdm

EMB = f"{ROOT}/embeddings"
device = "cuda"

def extract_clip(image_paths, out_path, batch_size=64):
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipping"); return
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    feats = np.zeros((len(image_paths), 768), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch = [Image.open(p).convert("RGB") for p in image_paths[i:i+batch_size]]
            inputs = proc(images=batch, return_tensors="pt").to(device)
            out = model.get_image_features(**inputs)  # (B, 768)
            feats[i:i+batch_size] = out.cpu().numpy()
    np.save(out_path, feats)
    del model; torch.cuda.empty_cache()

def extract_dinov2(image_paths, out_path, batch_size=64):
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipping"); return
    model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    feats = np.zeros((len(image_paths), 1024), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch = [Image.open(p).convert("RGB") for p in image_paths[i:i+batch_size]]
            inputs = proc(images=batch, return_tensors="pt").to(device)
            out = model(**inputs)
            # Use CLS token from last hidden state
            cls = out.last_hidden_state[:, 0, :]  # (B, 1024)
            feats[i:i+batch_size] = cls.cpu().numpy()
    np.save(out_path, feats)
    del model; torch.cuda.empty_cache()

# Build full path lists from the JSON manifests
with open(f"{PROC}/image_filenames_train.json") as f: train_rel = json.load(f)
with open(f"{PROC}/image_filenames_test.json") as f: test_rel = json.load(f)
train_paths = [f"{ROOT}/raw/thingseeg2_metadata/training_images/{p}" for p in train_rel]
test_paths  = [f"{ROOT}/raw/thingseeg2_metadata/test_images/{p}" for p in test_rel]

extract_clip(train_paths, f"{EMB}/clip_vitl14_train.npy")
extract_clip(test_paths,  f"{EMB}/clip_vitl14_test.npy")
extract_dinov2(train_paths, f"{EMB}/dinov2_large_train.npy")
extract_dinov2(test_paths,  f"{EMB}/dinov2_large_test.npy")
```

### Sanity checks

- All 4 `.npy` files exist with expected shapes
- No NaN, no all-zero rows
- Cosine similarity between embeddings of two images of the same THINGS concept (e.g. two `aardvark` images in the training set) should be higher than between random pairs. Print a few examples.

**Checkpoint:** `embeddings/` contains 4 files totaling ~150 MB. This phase never needs to run again.

---

## Phase 4 — Sanity-check baseline (ridge regression)

Goal: before building any transformer, verify the pipeline works end-to-end with a dead-simple linear model. This is your "the data is intact and correctly aligned" gate.

```python
from sklearn.linear_model import Ridge
import numpy as np

clip_train = np.load(f"{EMB}/clip_vitl14_train.npy")  # (16540, 768)
clip_test  = np.load(f"{EMB}/clip_vitl14_test.npy")   # (200, 768)

results = {}
for i in range(1, 11):
    sub = f"sub-{i:02d}"
    eeg_train = np.load(f"{PROC}/eeg_train_avg_{sub}.npz")["eeg"]  # (16540, C, T)
    eeg_test  = np.load(f"{PROC}/eeg_test_avg_{sub}.npz")["eeg"]   # (200, C, T)
    C, T = eeg_train.shape[1], eeg_train.shape[2]
    y_train = eeg_train.reshape(16540, C * T)
    y_test  = eeg_test.reshape(200, C * T)
    model = Ridge(alpha=1e5)  # tune later, but this is a reasonable start
    model.fit(clip_train, y_train)
    pred = model.predict(clip_test)
    # Pearson R per (channel, timepoint), then average
    pred_r = pred.reshape(200, C, T)
    true_r = y_test.reshape(200, C, T)
    r_per_ct = np.zeros((C, T))
    for c in range(C):
        for t in range(T):
            r_per_ct[c, t] = np.corrcoef(pred_r[:, c, t], true_r[:, c, t])[0, 1]
    results[sub] = {"mean_r": float(np.nanmean(r_per_ct)),
                    "max_r": float(np.nanmax(r_per_ct))}

import json
with open(f"{ROOT}/results/phase4_ridge_baseline.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Success criterion

Mean Pearson R across subjects should be in the range ~0.10–0.25. If you're getting near-zero, the alignment between embeddings and EEG is broken — almost always the image-filename ordering. Stop and debug.

If you're getting >0.4, that's suspicious — probably a leak (e.g., test images in training set, or test reps included in training averaging).

**Checkpoint:** `results/phase4_ridge_baseline.json` exists. The user has confirmation the pipeline works.

---

## Phase 5 — The three models

Goal: implement and train the three architectures from the plan, save weights to Drive.

### Architecture spec (multi-subject transformer)

```
Input:
  - image_emb: (B, D_emb)  where D_emb = 768 (CLIP) or 1024 (DINOv2)
  - context_embs: (B, K, D_emb)  K = 10 surrounding RSVP trials (5 before, 5 after)
                                  zero-padded for boundary cases
  - subject_id: (B,)  integer in [0, 10)

Stem:
  - Linear projection: D_emb → d_model (d_model = 384)
  - LayerNorm

Transformer encoder:
  - 4 layers, 6 heads, d_model = 384, ffn_dim = 1536
  - Learned positional embeddings for the (1 + K) sequence positions
  - The "current" token (position 0) is the one we read out

Subject block:
  - For each subject s, a linear layer W_s: (d_model) → (C * T)
  - Implemented as a (n_subjects, d_model, C * T) parameter, indexed by subject_id
  - For 10 subjects × 384 × 17×100 = ~6.5M params (manageable)

Output: (B, C, T)  predicted EEG epoch
Loss: MSE
```

### Training config

- Optimizer: AdamW, lr 1e-4, weight decay 0.01
- Schedule: cosine, 5% linear warmup
- Batch size: 64
- Epochs: 30 (with early stopping on val Pearson R, patience 5)
- Validation split: hold out 10% of training images (concept-level split — never put two images of the same concept in different splits)
- Save best checkpoint by val Pearson R to `checkpoints/multi_subject_transformer/best.pt`
- Append per-epoch metrics to `logs/multi_subject_transformer.jsonl`

### The three runs

Each run trains, saves checkpoints, evaluates on test set, writes results JSON.

**Run 1 — Per-subject linear baseline** (formalized as PyTorch for fair comparison)
- Single linear layer: D_emb → C*T
- Trained independently per subject, no context window
- 10 separate model files in `checkpoints/linear_baseline/`

**Run 2 — Per-subject transformer**
- Same architecture as multi-subject, but:
  - No subject block; final layer is just `Linear(d_model, C*T)`
  - Trained independently for each subject
- 10 separate model files in `checkpoints/per_subject_transformer/`

**Run 3 — Multi-subject transformer with subject block**
- Full architecture as specified above
- Trained jointly on all 10 subjects
- One model file in `checkpoints/multi_subject_transformer/`

### RSVP context window construction

The context for image `i` is the 5 images that preceded it in the RSVP stream and the 5 that followed, in their original presentation order. The presentation order is given by the trial order in the original EEG file — you'll need to extract this from the metadata, or use the natural sequential order of the array index (which, for THINGS-EEG2, *is* the presentation order within each session). Pad with zeros at the boundaries of each session.

> Be explicit in the log about what you used as the context-ordering source. This is the architectural piece that's genuinely novel to your project, so it's worth documenting clearly.

### Wall-clock targets (on H100)

- Linear baseline: ~5 min total for all 10 subjects
- Per-subject transformer: ~30–45 min total for all 10 subjects
- Multi-subject transformer: ~30–60 min for one run

Total H100 budget for Phase 5 with CLIP: ~1.5 hours. Then re-run multi-subject transformer with DINOv2 embeddings: another ~30–60 min. So Phase 5 total is ~2 hours of H100. Leaves you ~2 hours of H100 budget for Phase 7.

### Sanity checks

- Multi-subject transformer val loss should monotonically decrease (with normal stochastic wiggles) for at least the first 10 epochs
- Final test Pearson R, multi-subject transformer > per-subject transformer > linear baseline. If this ordering is violated, something is wrong — investigate before moving on.

**Checkpoint:** all three model variants saved to Drive. Per-epoch logs saved.

---

## Phase 6 — Evaluation and figures

Goal: produce the figures that go into the report.

### Figure 1 — Headline box plot (mirror of TRIBE Figure 2D)

Three columns: linear baseline, per-subject transformer, multi-subject transformer.
One dot per subject (10 dots per column).
Y-axis: mean Pearson R averaged across (channels × timepoints).
Connect each subject's three dots with thin gray lines to show within-subject improvements.
Paired t-tests with FDR correction across the three pairwise comparisons.

Save to `figures/headline_boxplot.png` (also `.pdf`).

### Figure 2 — Channel × timepoint correlation heatmap

For the multi-subject transformer (CLIP variant), compute Pearson R at each (channel, timepoint) cell, averaged across subjects and test images. Plot as a heatmap with channels on the y-axis (sorted by anatomical position — occipital electrodes at top), time on the x-axis (0 to 1000 ms post-stimulus).

You should see a hot spot in occipital channels around 100–200 ms (P1/N170) and weaker but persistent prediction in parietal channels through ~400 ms.

Save to `figures/channel_time_heatmap.png`.

### Figure 3 — CLIP vs DINOv2 ablation

Box plot, two columns. Same multi-subject architecture, swapped embedding source. Tells the reader whether self-supervised features matter for this task.

Save to `figures/embedding_ablation.png`.

### Results table

CSV with columns: `model`, `embedding`, `subject`, `mean_r`, `max_r`.
Save to `results/all_results.csv`.

---

## Phase 7 — Alljoined cross-hardware transfer

Goal: take the trained multi-subject transformer (CLIP variant, best checkpoint) and test how it transfers to the consumer-grade Emotiv hardware in Alljoined-1.6M. Run three variants.

### 7a. Download Alljoined

Alljoined-1.6M is the consumer-grade EEG dataset paired with THINGS images. At the time of writing it's hosted on HuggingFace; search for "Alljoined-1.6M" or "alljoined" and use the most recent canonical release. Confirm the link with the user before downloading. Save preprocessed epochs and image-trial mapping to `raw/alljoined/`.

### 7b. Channel intersection

Alljoined uses 32 Emotiv channels. THINGS-EEG2 uses 17 (or 63) BioSemi channels. Find the channel name intersection — likely 8–15 channels using the standard 10-20 system labels. Subset both datasets to the intersection.

This means: retrain the **multi-subject transformer with the channel-subset output** on THINGS-EEG2 first (a quick re-run of Phase 5 with a smaller output head). Save as `checkpoints/multi_subject_transformer/best_channelsubset.pt`. This is the transferable model.

### 7c. Three transfer variants

> Doing this on H100 means you can iterate same-day rather than waiting overnight on Colab. Budget ~2 hours of H100 for these three runs combined.

**Variant A — Zero-shot.** Take the channel-subset model. For each Alljoined subject, run inference with the "unseen subject" trick (you'll need to add a default unseen-subject pathway to the subject block during the original Phase 5 training — see TRIBE's section 5.3 on subject dropout). Compute test Pearson R. Save to `results/alljoined_zero_shot.json`.

**Variant B — Subject-block-only fine-tuning.** Freeze everything except the subject block. Initialize a new subject block with `n_subjects = n_alljoined_subjects`. Use low-rank initialization (rank 128) as TRIBE does. Train for 1 epoch on half of each Alljoined subject's data. Evaluate on the held-out half. Save to `results/alljoined_subject_block_ft.json`.

**Variant C — Full fine-tuning, low LR.** Same setup as Variant B but unfreeze everything, lr = 1e-5 (10× lower than original training). Train for 1 epoch. Save to `results/alljoined_full_ft.json`.

### 7d. Comparison baseline

For an honest comparison, also train a **per-subject linear ridge** from CLIP → Alljoined EEG, separately for each Alljoined subject. This is the fair "what you'd get without the transfer" baseline.

### 7e. The headline result

Report: "Zero-shot transfer of the THINGS-EEG2-trained model to Alljoined achieves R = X, vs R = Y for a per-subject ridge trained from scratch on Alljoined. With subject-block-only fine-tuning, R rises to Z." If X > Y or Z >> Y, you have evidence that cross-hardware transfer of TRIBE's recipe works.

Make a fourth figure: bar chart with bars for [Alljoined ridge from scratch, Variant A, Variant B, Variant C], one bar per condition, error bars across Alljoined subjects. Save to `figures/alljoined_transfer.png`.

---

## Phase 8 — Report assembly

Goal: a five-page report following the syllabus rubric (intro, related work, methodology, conclusions, future work).

Pull all figures from `figures/`, all numbers from `results/`. The report writes itself once the experiments are done — outline:

1. **Introduction (~0.75 pg).** TRIBE v2's claim, motivation for testing it on EEG, hypothesis statement.
2. **Related work (~0.75 pg).** TRIBE v2, THINGS-EEG2, ENIGMA, Alljoined, Antonello scaling laws.
3. **Methods (~1.5 pg).** Datasets, preprocessing, embedding extraction, the three architectures, the RSVP context window (highlight as your novel piece), evaluation protocol.
4. **Results (~1.5 pg).** Headline box plot, channel-time heatmap, embedding ablation, Alljoined transfer.
5. **Discussion + future work (~0.5 pg).** What worked, what didn't, what the next 6 months of this project would look like (full multi-dataset unification, source localization, scaling laws).

---

## Operational rules for Claude Code

- **Always check for existing checkpoints before recomputing.** Every phase script starts by checking whether the output already exists.
- **Print-and-pause pattern for big operations.** Before downloading >1 GB, before starting a training run >10 min, print what you're about to do and how long it'll take, and let the user `# continue` if needed.
- **All paths are absolute** and start with `/content/drive/MyDrive/tribe-eeg/` or `/content/work/`. Never use relative paths in saved scripts.
- **Log everything to `{ROOT}/logs/`.** Each phase has its own JSONL log with timestamped events.
- **If you hit an unexpected file structure, stop and surface it.** The most common failure mode in this project is the THINGS-EEG2 preprocessed file layout being subtly different from what's described above. Inspect first, write code second.
- **Random seeds.** Set `torch.manual_seed(42)` and `np.random.seed(42)` at the top of every training script. For seed-averaging in Phase 5, use seeds [42, 1337, 2026].
- **GPU memory.** If batch size is too large, halve it and warn the user. Don't silently OOM.

---

## Compute budget summary (5 H100-hours total)

| Task | H100 time |
|---|---|
| Phase 3: CLIP + DINOv2 embeddings | ~20 min |
| Phase 4: ridge sanity check | negligible (CPU) |
| Phase 5 with CLIP (3 models, 1 seed each) | ~1.5 hr |
| Phase 5 with DINOv2 (just the multi-subject run) | ~45 min |
| Phase 5 seed averaging (2 extra seeds × multi-subject CLIP) | ~1 hr |
| Phase 7: Alljoined transfer (3 variants + ridge baseline) | ~1.5 hr |
| Buffer | ~30 min |
| **Total** | ~5 hours |

If H100 isn't available in a given session, downgrade gracefully: training runs go from ~30 min to ~3 hours on a T4. The pipeline still works; iteration is just slower.

---

## Start condition

Begin with Phase 0. Before any phase that takes >5 minutes, confirm with the user. After every phase, write a short status summary to chat: what was done, where the artifacts are, what's next.

If you have any questions about interpretation before starting, ask them all at once in your first reply.