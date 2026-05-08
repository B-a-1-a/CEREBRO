import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def compute_pearson_r(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson R per (channel, timepoint) across N test samples.

    Args:
        pred:   (N, C, T) float32
        target: (N, C, T) float32
    Returns:
        r_matrix: (C, T) float32
    """
    p = pred   - pred.mean(0, keepdims=True)
    t = target - target.mean(0, keepdims=True)
    num = (p * t).sum(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0)).clip(min=1e-8)
    return (num / den).astype(np.float32)


def plot_channel_time_heatmap(r_matrix: np.ndarray, channel_names: list,
                               save_path: Optional[str] = None):
    """Channel x timepoint heatmap of Pearson R values.

    Args:
        r_matrix:      (C, T) float32
        channel_names: list of C channel name strings
        save_path:     optional path to save figure (PNG/PDF)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(r_matrix, aspect="auto", origin="upper",
                   cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    plt.colorbar(im, ax=ax, label="Pearson R")
    ax.set_yticks(range(len(channel_names)))
    ax.set_yticklabels(channel_names, fontsize=6)
    ax.set_xlabel("Time (sample index)")
    ax.set_title("Pearson R per channel x timepoint")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return fig
