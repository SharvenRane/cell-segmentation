import numpy as np
import torch

from src.data import make_blob_sample
from src.metrics import dice_coefficient
from src.model import UNet
from src.postprocess import count_instances, watershed_instances
from src.train import train_overfit


def test_unet_output_shape_matches_input():
    model = UNet(in_channels=1, out_channels=1, base_features=8)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_unet_runs_on_odd_friendly_size():
    model = UNet(in_channels=1, out_channels=1, base_features=8)
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    assert y.shape == (1, 1, 32, 32)
    assert torch.isfinite(y).all()


def test_training_reduces_loss():
    result = train_overfit(steps=60, lr=1e-2, seed=0)
    assert result["final_loss"] < result["initial_loss"]
    # After overfitting a tiny easy batch the loss should be clearly low.
    assert result["final_loss"] < 0.2


def test_trained_model_predicts_high_dice_on_easy_mask():
    result = train_overfit(steps=80, lr=1e-2, seed=0)
    model = result["model"]
    model.eval()

    rng = np.random.default_rng(0)
    sample = make_blob_sample(size=64, n_cells=3, noise=0.0, rng=rng)
    x = torch.from_numpy(sample.image).view(1, 1, 64, 64)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].numpy()
    pred = (prob > 0.5).astype(np.uint8)

    dice = dice_coefficient(pred, sample.mask)
    assert dice > 0.9


def test_end_to_end_pipeline_counts_cells():
    # Predict mask with a trained model, then split with watershed and count.
    result = train_overfit(steps=80, lr=1e-2, seed=0)
    model = result["model"]
    model.eval()

    rng = np.random.default_rng(3)
    sample = make_blob_sample(
        size=96, n_cells=3, radius_range=(8, 9), noise=0.0, rng=rng
    )
    x = torch.from_numpy(sample.image).view(1, 1, 96, 96)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].numpy()
    pred_mask = (prob > 0.5).astype(np.uint8)

    labels = watershed_instances(pred_mask, min_distance=5)
    true_count = count_instances(sample.instances)
    # The recovered count should be at least close to the truth.
    assert abs(count_instances(labels) - true_count) <= 1
