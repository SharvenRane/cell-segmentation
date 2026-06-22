import numpy as np

from src.data import make_blob_sample, make_two_touching_circles
from src.postprocess import count_instances, watershed_instances


def test_watershed_splits_two_touching_circles():
    sample = make_two_touching_circles(size=64, radius=14, gap=4, noise=0.0)
    # The binary mask is a single connected blob.
    from scipy import ndimage as ndi

    _, n_components = ndi.label(sample.mask)
    assert n_components == 1

    labels = watershed_instances(sample.mask, min_distance=5)
    assert count_instances(labels) == 2


def test_watershed_empty_mask_returns_zero_instances():
    empty = np.zeros((32, 32), dtype=np.uint8)
    labels = watershed_instances(empty)
    assert count_instances(labels) == 0
    assert labels.shape == empty.shape


def test_watershed_single_circle_is_one_instance():
    mask = np.zeros((48, 48), dtype=np.uint8)
    yy, xx = np.ogrid[:48, :48]
    mask[(yy - 24) ** 2 + (xx - 24) ** 2 <= 12 ** 2] = 1
    labels = watershed_instances(mask, min_distance=5)
    assert count_instances(labels) == 1


def test_watershed_recovers_separated_blobs():
    rng = np.random.default_rng(7)
    # Well separated cells so watershed should find each one.
    sample = make_blob_sample(
        size=96, n_cells=3, radius_range=(7, 8), noise=0.0, rng=rng
    )
    labels = watershed_instances(sample.mask, min_distance=5)
    # Ground truth count is the number of distinct ids actually painted.
    true_count = count_instances(sample.instances)
    assert count_instances(labels) == true_count


def test_labels_are_subset_of_mask():
    sample = make_two_touching_circles(size=64, radius=14, gap=4)
    labels = watershed_instances(sample.mask)
    # No label should appear where the mask is background.
    assert np.all(labels[sample.mask == 0] == 0)
