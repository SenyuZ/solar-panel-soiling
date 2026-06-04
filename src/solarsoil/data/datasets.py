"""Torch datasets, transforms and dataloaders driven by a manifest CSV.

Transforms mirror the original notebook (ImageNet normalisation, light
augmentation), with two fixes for robustness:

* the RGBA->RGB conversion is a named, *picklable* function (the notebook used a
  lambda, which breaks ``num_workers > 0`` on spawn-based platforms like macOS);
* a fixed, taxonomy-derived class ordering so label indices are stable and
  metrics are directly comparable across runs and label spaces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .. import taxonomy

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _to_rgb(img: Image.Image) -> Image.Image:
    """Flatten transparency and guarantee 3-channel RGB (picklable)."""
    return img.convert("RGBA").convert("RGB")


def build_transforms(img_size: int = 224, train: bool = True, augment: bool = True):
    """Return a torchvision transform pipeline.

    Training pipeline (when ``augment``) adds horizontal flip, small rotation and
    mild colour jitter; validation/test is deterministic resize + normalise.
    """
    from torchvision import transforms

    steps = [transforms.Lambda(_to_rgb), transforms.Resize((img_size, img_size))]
    if train and augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class ManifestDataset(Dataset):
    """Reads images for one split from a manifest produced by :mod:`manifest`."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        split: str,
        class_names: list[str],
        img_size: int = 224,
        train: bool = False,
        augment: bool = True,
        repo_root: str | Path | None = None,
    ) -> None:
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split={split!r} in manifest")
        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        unknown = set(self.df["label"]) - set(self.class_to_idx)
        if unknown:
            raise ValueError(f"Manifest labels {unknown} not in class_names {self.class_names}")
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.transform = build_transforms(img_size, train=train, augment=augment)

    def __len__(self) -> int:
        return len(self.df)

    def targets(self) -> list[int]:
        return [self.class_to_idx[l] for l in self.df["label"]]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(self.repo_root / row["filepath"]) as img:
            x = self.transform(img)
        y = self.class_to_idx[row["label"]]
        return x, y


def class_weights(counts: np.ndarray):
    """Inverse-frequency class weights, normalised to sum to 1 (as in the notebook)."""
    import torch

    counts = np.asarray(counts, dtype=np.float64)
    w = 1.0 / np.clip(counts, 1, None)
    w = w / w.sum()
    return torch.tensor(w, dtype=torch.float32)


def make_dataloaders(
    manifest: str | Path,
    label_space: str = "binary",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    repo_root: str | Path | None = None,
    pin_memory: bool = False,
) -> dict:
    """Build train/val/test loaders + metadata from a manifest CSV.

    Returns a dict with ``loaders`` (per available split), ``class_names``,
    ``class_counts`` (train) and ``class_weights`` (train, inverse-frequency).
    """
    df = pd.read_csv(manifest)
    class_names = taxonomy.label_space(label_space)

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        if split not in set(df["split"]):
            continue
        ds = ManifestDataset(
            df,
            split=split,
            class_names=class_names,
            img_size=img_size,
            train=(split == "train"),
            augment=augment and split == "train",
            repo_root=repo_root,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    train_df = df[df["split"] == "train"]
    counts = np.array([(train_df["label"] == c).sum() for c in class_names])
    return {
        "loaders": loaders,
        "class_names": class_names,
        "class_counts": counts,
        "class_weights": class_weights(counts),
    }
