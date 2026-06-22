"""Watershed post processing to split touching cells into instances.

A binary foreground mask only tells us where cells are, not how many.
When two cells touch, the mask is a single blob. The distance transform of
the mask peaks near each cell center, so seeding watershed at those peaks
and letting it flood the foreground separates the blob along the thin neck
between cells.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def watershed_instances(
    mask: np.ndarray,
    min_distance: int = 5,
    footprint_size: int = 3,
) -> np.ndarray:
    """Split a binary mask into instance labels with watershed.

    Args:
        mask: binary foreground array of shape (H, W). Nonzero is foreground.
        min_distance: minimum number of pixels between two distance peaks.
            This controls how aggressively touching cells are split.
        footprint_size: side length of the square neighborhood used when
            looking for local maxima in the distance transform.

    Returns:
        An int32 label map of shape (H, W). Background is 0 and each cell
        has a distinct positive id.
    """
    binary = np.asarray(mask) > 0
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(binary)

    footprint = np.ones((footprint_size, footprint_size), dtype=bool)
    peak_coords = peak_local_max(
        distance,
        min_distance=min_distance,
        footprint=footprint,
        labels=binary,
    )

    markers = np.zeros(distance.shape, dtype=np.int32)
    if peak_coords.shape[0] == 0:
        # No clear peak found, fall back to connected components so we still
        # return at least one instance for a nonempty mask.
        labeled, _ = ndi.label(binary)
        return labeled.astype(np.int32)

    for idx, (r, c) in enumerate(peak_coords, start=1):
        markers[r, c] = idx

    labels = watershed(-distance, markers, mask=binary)
    return labels.astype(np.int32)


def count_instances(labels: np.ndarray) -> int:
    """Count distinct positive ids in a label map."""
    arr = np.asarray(labels)
    unique = np.unique(arr)
    return int((unique > 0).sum())
