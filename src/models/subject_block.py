import torch
import torch.nn as nn


class SubjectBlock(nn.Module):
    """Per-subject output projection. Parameter matrix (n_subjects+1, d_model, C*T).
    The +1 slot is the unknown-subject token used during subject dropout.

    Subject dropout (p=dropout_p) replaces subject_id with the unknown slot during
    training -- required for Phase 7 zero-shot transfer to unseen subjects.
    """

    def __init__(self, n_subjects: int, d_model: int, n_ch: int, n_t: int,
                 dropout_p: float = 0.1):
        super().__init__()
        self.n_subjects = n_subjects
        self.dropout_p  = dropout_p
        self.n_ch = n_ch
        self.n_t  = n_t
        self.weight = nn.Parameter(torch.randn(n_subjects + 1, d_model, n_ch * n_t) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(n_subjects + 1, n_ch * n_t))

    def forward(self, x: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model), subject_idx: (B,) int
        if self.training and self.dropout_p > 0:
            mask = torch.rand(x.shape[0], device=x.device) < self.dropout_p
            subject_idx = subject_idx.clone()
            subject_idx[mask] = self.n_subjects
        W   = self.weight[subject_idx]                          # (B, d_model, C*T)
        b   = self.bias[subject_idx]                            # (B, C*T)
        out = torch.bmm(x.unsqueeze(1), W).squeeze(1) + b      # (B, C*T)
        return out.view(-1, self.n_ch, self.n_t)
