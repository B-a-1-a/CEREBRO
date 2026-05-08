import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ThingsEEG2Dataset(Dataset):
    """
    THINGS-EEG2 averaged EEG responses paired with image embeddings.

    Each item returns (image_emb, context_embs, subject_idx, eeg):
      image_emb:    (D_emb,)    current image embedding
      context_embs: (K, D_emb)  ±K/2 RSVP-adjacent embeddings, zero-padded at boundaries
      subject_idx:  scalar int  index into self.subjects
      eeg:          (C, T)      averaged EEG epoch
    """

    def __init__(self, root, split="train", subjects=None, embedding="clip_vitl14", context_k=10):
        """
        Args:
            root:       path to tribe-eeg/ on Drive
            split:      'train' or 'test'
            subjects:   list of 1-indexed subject ints, or None for all 10
            embedding:  'clip_vitl14' or 'dinov2_large'
            context_k:  total context window size (context_k/2 before, context_k/2 after)
        """
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split}"
        assert context_k % 2 == 0, "context_k must be even"

        self.root = root
        self.split = split
        self.context_k = context_k

        if subjects is None:
            subjects = list(range(1, 11))
        self.subjects = subjects
        self.subject_to_idx = {s: i for i, s in enumerate(subjects)}

        emb_path = os.path.join(root, "embeddings", f"{embedding}_{split}.npy")
        self.embeddings = np.load(emb_path).astype(np.float32)  # (N, D_emb)
        self.n_images, self.D_emb = self.embeddings.shape

        self.eeg = {}
        for s in subjects:
            sub_id = f"sub-{s:02d}"
            npz_path = os.path.join(root, "processed", f"eeg_{split}_avg_{sub_id}.npz")
            self.eeg[s] = np.load(npz_path)["eeg"].astype(np.float32)  # (N, C, T)

        self.index = [(s, i) for s in subjects for i in range(self.n_images)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        subject_id, img_idx = self.index[idx]

        image_emb = self.embeddings[img_idx]

        half = self.context_k // 2
        context_embs = np.zeros((self.context_k, self.D_emb), dtype=np.float32)
        pos = 0
        for offset in range(-half, half + 1):
            if offset == 0:
                continue
            neighbor = img_idx + offset
            if 0 <= neighbor < self.n_images:
                context_embs[pos] = self.embeddings[neighbor]
            pos += 1

        eeg = self.eeg[subject_id][img_idx]
        subject_idx = self.subject_to_idx[subject_id]

        return (
            torch.from_numpy(image_emb),
            torch.from_numpy(context_embs),
            torch.tensor(subject_idx, dtype=torch.long),
            torch.from_numpy(eeg),
        )

    @property
    def eeg_shape(self):
        s = self.subjects[0]
        return self.eeg[s].shape[1:]  # (C, T)
