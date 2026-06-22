"""Segmentation metrics: Dice for masks and an instance count error.

Dice measures how well a predicted binary mask overlaps the ground truth.
The instance count error compares how many cells we recover after the
watershed split against the true number of cells.
"""

from __future__ import annotations

import numpy as np
import torch


def dice_coefficient(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    eps: float = 1e-7,
) -> float:
    """Dice overlap between two binary masks.

    Both inputs are treated as binary by thresholding at 0.5. Accepts numpy
    arrays or torch tensors of any matching shape.

    Args:
        pred: predicted mask.
        target: ground truth mask.
        eps: smoothing term that keeps the score defined when both masks
            are empty (in which case the score is 1.0).

    Returns:
        Dice score in [0, 1].
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {t.shape}")

    p_bin = (p > 0.5).astype(np.float64)
    t_bin = (t > 0.5).astype(np.float64)

    intersection = float((p_bin * t_bin).sum())
    denom = float(p_bin.sum() + t_bin.sum())
    return (2.0 * intersection + eps) / (denom + eps)


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable soft Dice loss on logits.

    Args:
        logits: raw network output of shape (N, 1, H, W) or (N, H, W).
        target: binary target of the same broadcastable shape.
        eps: smoothing term.

    Returns:
        A scalar tensor, 1 minus the soft Dice score.
    """
    probs = torch.sigmoid(logits)
    target = target.to(probs.dtype)
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def instance_count_error(pred_count: int, true_count: int) -> int:
    """Absolute difference between predicted and true instance counts."""
    return abs(int(pred_count) - int(true_count))


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
