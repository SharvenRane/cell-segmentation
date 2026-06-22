"""Synthetic blob generator for cell segmentation experiments.

Each sample is a grayscale image with a handful of bright circular blobs
on a dark background, paired with a binary foreground mask and an integer
instance label map. The blobs may touch, which is exactly the case the
watershed step is meant to handle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    """A single synthetic sample.

    Attributes:
        image: float32 array of shape (H, W) with values in [0, 1].
        mask: uint8 binary foreground mask of shape (H, W).
        instances: int32 label map of shape (H, W); 0 is background and
            each cell has a distinct positive id.
    """

    image: np.ndarray
    mask: np.ndarray
    instances: np.ndarray


def _disk_mask(h: int, w: int, cy: float, cx: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2


def make_blob_sample(
    size: int = 64,
    n_cells: int = 4,
    radius_range: tuple[int, int] = (6, 10),
    noise: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Sample:
    """Generate one synthetic image with circular cells.

    Args:
        size: image height and width in pixels.
        n_cells: number of cells to place.
        radius_range: inclusive range of cell radii.
        noise: standard deviation of additive Gaussian noise on the image.
        rng: optional numpy random generator for reproducibility.

    Returns:
        A Sample with image, binary mask, and instance label map.
    """
    if rng is None:
        rng = np.random.default_rng()

    image = np.zeros((size, size), dtype=np.float32)
    instances = np.zeros((size, size), dtype=np.int32)

    next_id = 1
    for _ in range(n_cells):
        radius = int(rng.integers(radius_range[0], radius_range[1] + 1))
        cy = float(rng.integers(radius, size - radius))
        cx = float(rng.integers(radius, size - radius))
        disk = _disk_mask(size, size, cy, cx, radius)
        image[disk] = 1.0
        # Later cells overwrite earlier ids where they overlap so that the
        # label map stays a clean partition of the foreground.
        instances[disk] = next_id
        next_id += 1

    if noise > 0:
        image = image + rng.normal(0.0, noise, image.shape).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

    mask = (instances > 0).astype(np.uint8)
    return Sample(image=image, mask=mask, instances=instances)


def make_two_touching_circles(
    size: int = 64,
    radius: int = 14,
    gap: int = 4,
    noise: float = 0.0,
) -> Sample:
    """Generate two horizontally adjacent circles whose disks touch.

    The two circles share a thin bridge of foreground pixels so the binary
    mask is a single connected component. Watershed should still recover
    two instances.

    Args:
        size: image height and width.
        radius: radius of each circle.
        gap: center separation beyond ``2 * radius`` is negative when the
            circles overlap. The centers are placed at distance
            ``2 * radius - gap`` so a positive ``gap`` makes them overlap.
        noise: optional additive Gaussian noise.

    Returns:
        A Sample. The instance map holds the ground truth two cell split.
    """
    rng = np.random.default_rng(0)
    cy = size / 2.0
    sep = 2 * radius - gap
    cx_left = size / 2.0 - sep / 2.0
    cx_right = size / 2.0 + sep / 2.0

    left = _disk_mask(size, size, cy, cx_left, radius)
    right = _disk_mask(size, size, cy, cx_right, radius)

    instances = np.zeros((size, size), dtype=np.int32)
    instances[left] = 1
    # Right disk wins on the overlap so each pixel has one ground truth id.
    instances[right] = 2

    image = (left | right).astype(np.float32)
    if noise > 0:
        image = image + rng.normal(0.0, noise, image.shape).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

    mask = (left | right).astype(np.uint8)
    return Sample(image=image, mask=mask, instances=instances)
