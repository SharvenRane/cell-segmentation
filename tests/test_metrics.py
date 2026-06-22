import numpy as np
import torch

from src.metrics import dice_coefficient, instance_count_error, soft_dice_loss


def test_dice_identical_masks_is_one():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[4:12, 4:12] = 1
    assert dice_coefficient(m, m) == 1.0


def test_dice_disjoint_masks_is_zero():
    a = np.zeros((16, 16), dtype=np.uint8)
    b = np.zeros((16, 16), dtype=np.uint8)
    a[0:4, 0:4] = 1
    b[10:14, 10:14] = 1
    assert dice_coefficient(a, b) < 1e-3


def test_dice_half_overlap_is_two_thirds():
    # Two equal area boxes that share exactly half their pixels give
    # Dice = 2 * 0.5 / (1 + 1) ... here intersection 8, sizes 16 and 16.
    a = np.zeros((4, 8), dtype=np.uint8)
    b = np.zeros((4, 8), dtype=np.uint8)
    a[:, 0:4] = 1  # 16 pixels
    b[:, 2:6] = 1  # 16 pixels, overlap on columns 2,3 => 8 pixels
    score = dice_coefficient(a, b)
    assert abs(score - (2 * 8) / (16 + 16)) < 1e-6


def test_dice_both_empty_is_one():
    z = np.zeros((8, 8), dtype=np.uint8)
    assert dice_coefficient(z, z) == 1.0


def test_dice_accepts_torch_tensors():
    t = torch.zeros(8, 8)
    t[2:6, 2:6] = 1.0
    assert dice_coefficient(t, t) == 1.0


def test_soft_dice_loss_is_low_for_confident_correct():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    logits = torch.where(target > 0.5, 10.0, -10.0)
    loss = soft_dice_loss(logits, target)
    assert float(loss) < 0.05


def test_soft_dice_loss_gradient_flows():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    loss = soft_dice_loss(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_instance_count_error():
    assert instance_count_error(3, 3) == 0
    assert instance_count_error(2, 4) == 2
