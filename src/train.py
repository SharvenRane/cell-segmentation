"""A tiny training loop for the U-Net on synthetic blobs.

This is deliberately small. It overfits a handful of synthetic samples so
that the network learns to reproduce easy masks. It is enough to show the
pieces fit together and to support a behavior test that the loss drops.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import make_blob_sample
from .metrics import soft_dice_loss
from .model import UNet


def make_batch(
    n: int = 4,
    size: int = 64,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a batch of (image, mask) tensors from synthetic blobs.

    Returns:
        images of shape (n, 1, size, size) and masks of shape (n, 1, size, size).
    """
    rng = np.random.default_rng(seed)
    imgs = []
    masks = []
    for _ in range(n):
        s = make_blob_sample(size=size, n_cells=3, noise=0.0, rng=rng)
        imgs.append(s.image)
        masks.append(s.mask.astype(np.float32))
    images = torch.from_numpy(np.stack(imgs)).unsqueeze(1)
    targets = torch.from_numpy(np.stack(masks)).unsqueeze(1)
    return images, targets


def train_overfit(
    steps: int = 60,
    lr: float = 1e-2,
    seed: int = 0,
) -> dict[str, float | UNet]:
    """Overfit the U-Net to a fixed synthetic batch.

    Args:
        steps: number of optimizer steps.
        lr: learning rate for Adam.
        seed: seed for batch generation and weight init.

    Returns:
        A dict with the initial loss, final loss, and the trained model.
    """
    torch.manual_seed(seed)
    model = UNet(in_channels=1, out_channels=1, base_features=16)
    images, targets = make_batch(n=4, size=64, seed=seed)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    initial_loss = None
    loss_value = 0.0
    for step in range(steps):
        opt.zero_grad()
        logits = model(images)
        loss = soft_dice_loss(logits, targets)
        loss.backward()
        opt.step()
        loss_value = float(loss.item())
        if step == 0:
            initial_loss = loss_value

    return {
        "initial_loss": float(initial_loss if initial_loss is not None else loss_value),
        "final_loss": loss_value,
        "model": model,
    }
