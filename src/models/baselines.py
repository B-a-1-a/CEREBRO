import torch
import torch.nn as nn


class PerSubjectLinearBaseline(nn.Module):
    """Single linear layer: CLIP embedding -> predicted EEG. Trained per subject."""

    def __init__(self, d_emb: int = 768, n_ch: int = 63, n_t: int = 100):
        super().__init__()
        self.proj = nn.Linear(d_emb, n_ch * n_t)
        self.n_ch = n_ch
        self.n_t = n_t

    def forward(self, image_emb: torch.Tensor, context_embs=None, subject_idx=None):
        return self.proj(image_emb).view(-1, self.n_ch, self.n_t)
