import numpy as np
import torch
from PIL import Image

from solarsoil.models.segmentation import (
    SegDataset,
    UNet,
    build_seg_manifest,
    coverage_from_prob,
    dice_loss,
)


def test_unet_forward_shape():
    model = UNet(base=8)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)  # same spatial size, 1 logit channel


def test_dice_loss_in_range():
    logits = torch.randn(2, 1, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loss = dice_loss(logits, target)
    assert 0.0 <= loss.item() <= 1.0


def test_coverage_from_prob():
    prob = np.zeros((10, 10), dtype=np.float32)
    prob[:5, :] = 1.0  # half the pixels "dusty"
    assert abs(coverage_from_prob(prob) - 0.5) < 1e-6


def _make_pairs(root, n=12):
    rng = np.random.default_rng(0)
    imgs, masks = root / "images", root / "masks"
    imgs.mkdir()
    masks.mkdir()
    for i in range(n):
        Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8")).save(imgs / f"img_{i}.jpg")
        Image.fromarray((rng.random((32, 32)) * 255).astype("uint8")).save(masks / f"img_{i}.png")
    return imgs, masks


def test_build_seg_manifest_and_dataset(tmp_path):
    imgs, masks = _make_pairs(tmp_path)
    out = tmp_path / "seg.csv"
    df = build_seg_manifest(imgs, masks, out, val_size=0.25, test_size=0.25, repo_root=tmp_path)
    assert {"image", "mask", "split"}.issubset(df.columns)
    assert set(df["split"]) == {"train", "val", "test"}
    ds = SegDataset(df, "train", img_size=32, repo_root=tmp_path)
    x, m = ds[0]
    assert x.shape == (3, 32, 32)
    assert m.shape == (1, 32, 32)
