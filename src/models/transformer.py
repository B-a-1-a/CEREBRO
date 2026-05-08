import torch
import torch.nn as nn
from src.models.subject_block import SubjectBlock

D_MODEL  = 384
N_HEADS  = 6
N_LAYERS = 4
FFN_DIM  = 1536


class _TransformerBase(nn.Module):
    """Shared encoder backbone for per-subject and multi-subject variants."""

    def __init__(self, d_emb: int = 768, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS,
                 ffn_dim: int = FFN_DIM):
        super().__init__()
        self.input_proj = nn.Linear(d_emb, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)

    def encode(self, image_emb: torch.Tensor, context_embs: torch.Tensor) -> torch.Tensor:
        # image_emb: (B, D), context_embs: (B, K, D)
        x = torch.cat([image_emb.unsqueeze(1), context_embs], dim=1)  # (B, K+1, D)
        x = self.input_proj(x)
        ctx_pad = (context_embs.abs().sum(-1) == 0)             # (B, K) True=padding
        img_pad = torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([img_pad, ctx_pad], dim=1)         # (B, K+1)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x[:, 0]                                          # image-token output


class PerSubjectTransformer(_TransformerBase):
    """Transformer trained independently per subject with a shared linear output head."""

    def __init__(self, d_emb: int = 768, n_ch: int = 63, n_t: int = 100, **kw):
        super().__init__(d_emb=d_emb, **kw)
        self.head = nn.Linear(D_MODEL, n_ch * n_t)
        self.n_ch = n_ch
        self.n_t  = n_t

    def forward(self, image_emb, context_embs, subject_idx=None):
        return self.head(self.encode(image_emb, context_embs)).view(-1, self.n_ch, self.n_t)


class MultiSubjectTransformer(_TransformerBase):
    """Transformer trained jointly across all subjects with SubjectBlock output."""

    def __init__(self, d_emb: int = 768, n_ch: int = 63, n_t: int = 100,
                 n_subjects: int = 10, subject_dropout: float = 0.1, **kw):
        super().__init__(d_emb=d_emb, **kw)
        self.subject_block = SubjectBlock(n_subjects, D_MODEL, n_ch, n_t, subject_dropout)

    def forward(self, image_emb, context_embs, subject_idx):
        return self.subject_block(self.encode(image_emb, context_embs), subject_idx)
