"""Track B: soiling-severity regression on measured % power loss (DeepSolarEye).

DeepSolarEye image filenames encode the measured power loss and irradiance, e.g.
``solar_..._L_<powerloss>_I_<irradiance>.jpg``. This module parses that target,
builds a regression manifest, and fine-tunes a CNN with a single-output head and
MSE loss to predict power loss directly — the data-grounded counterpart to the
classical Track A soiling index in :mod:`solarsoil.severity`.

Usage::

    # 1. download DeepSolarEye (auto via gdown — see solarsoil.data.download)
    # 2. build the regression manifest by parsing power loss from filenames
    python -m solarsoil.models.regression manifest \
        --data-root Data/raw/deepsolareye --out manifests/deepsolareye_manifest.csv
    # 3. train the regressor
    python -m solarsoil.models.regression train --config configs/severity.yaml
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ..data.datasets import build_transforms
from ..data.dedup import iter_images
from ..engine import load_checkpoint, save_checkpoint
from ..models.cnn import build_model
from ..utils import get_device, load_config, save_json, set_seed, setup_logging

logger = logging.getLogger("solarsoil.regression")

# DeepSolarEye filenames: power loss follows "_L_" and precedes "_I_".
DEFAULT_LABEL_REGEX = r"_L_([-+0-9.]+)_I_"


def parse_target(filename: str, pattern: re.Pattern) -> float | None:
    m = pattern.search(filename)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def build_regression_manifest(
    data_root: str | Path,
    out_csv: str | Path,
    label_regex: str = DEFAULT_LABEL_REGEX,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 123,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    """Scan images, parse the regression target from each filename, split, write CSV."""
    repo_root = Path(repo_root) if repo_root else Path.cwd()
    pat = re.compile(label_regex)
    rows = []
    for img in iter_images(data_root):
        target = parse_target(img.name, pat)
        if target is None:
            continue
        rel = img.resolve().relative_to(repo_root.resolve()).as_posix()
        rows.append({"filepath": rel, "power_loss": target})
    if not rows:
        raise RuntimeError(f"No labelled images parsed under {data_root} with regex {label_regex!r}")

    df = pd.DataFrame(rows).reset_index(drop=True)
    tr, tmp = train_test_split(df.index, test_size=val_size + test_size, random_state=seed)
    va, te = train_test_split(tmp, test_size=test_size / (val_size + test_size), random_state=seed)
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(
        "Wrote %s (%d images; power_loss range %.3f–%.3f)",
        out_csv, len(df), df["power_loss"].min(), df["power_loss"].max(),
    )
    return df


class RegressionDataset(Dataset):
    """Images + a continuous target read from a regression manifest."""

    def __init__(self, manifest, split, img_size=224, train=False, augment=True, repo_root=None):
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split={split!r}")
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.transform = build_transforms(img_size, train=train, augment=augment)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.repo_root / row["filepath"]) as im:
            x = self.transform(im)
        return x, torch.tensor([float(row["power_loss"])], dtype=torch.float32)


def load_regressor(model_path: str | Path, device):
    """Load a trained Track B power-loss regressor bundle for inference."""
    bundle = load_checkpoint(model_path, map_location=device)
    model, _ = build_model(bundle.get("backbone", "resnet50"), num_classes=1, pretrained=False)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device).eval()
    return model, bundle


@torch.no_grad()
def predict_power_loss(model, bundle, image, device) -> float:
    """Predict the measured power-loss *fraction* (clamped to [0, 1]) for one image.

    ``image`` may be a PIL image or a path. Multiply by 100 for a percentage.
    """
    tfm = build_transforms(int(bundle.get("img_size", 224)), train=False)
    if isinstance(image, Image.Image):
        x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
    else:
        with Image.open(image) as im:
            x = tfm(im.convert("RGB")).unsqueeze(0).to(device)
    pred = float(model(x).item())
    return max(0.0, min(1.0, pred))


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total, n, preds, tgts = 0.0, 0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            preds.append(out.detach().cpu().numpy())
            tgts.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds).ravel()
    tgts = np.concatenate(tgts).ravel()
    mae = float(np.mean(np.abs(preds - tgts)))
    rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
    return total / max(n, 1), mae, rmse


def train_regressor(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 123)))
    device = get_device(cfg.get("device", "auto"))
    repo_root = Path(cfg.get("repo_root", Path.cwd()))
    out_dir = Path(cfg.get("out_dir", "artifacts/severity"))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = cfg["manifest"]
    img_size = int(cfg.get("img_size", 224))

    df = pd.read_csv(manifest)
    if cfg.get("limit"):
        limit = int(cfg["limit"])
        seed = int(cfg.get("seed", 123))
        frac = min(1.0, limit / len(df))
        # Stratified subsample: keep the train/val/test proportions.
        df = pd.concat(
            [g.sample(frac=frac, random_state=seed) for _, g in df.groupby("split")]
        ).reset_index(drop=True)
        logger.info("Subsampled to %d images (stratified by split): %s",
                    len(df), df["split"].value_counts().to_dict())
    loaders = {}
    for split in ("train", "val", "test"):
        if split not in set(df["split"]):
            continue
        ds = RegressionDataset(df, split, img_size, train=(split == "train"),
                               augment=(split == "train"), repo_root=repo_root)
        loaders[split] = DataLoader(ds, batch_size=int(cfg.get("batch_size", 64)),
                                    shuffle=(split == "train"), num_workers=int(cfg.get("num_workers", 2)))

    model, _ = build_model(cfg.get("backbone", "resnet50"), num_classes=1, pretrained=cfg.get("pretrained", True))
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr_head", 1e-4)),
                                 weight_decay=float(cfg.get("weight_decay", 0.0)))

    epochs = int(cfg.get("epochs", 25))
    patience = int(cfg.get("patience", 5))
    best, wait = float("inf"), 0
    for ep in range(epochs):
        tr = _run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        va = _run_epoch(model, loaders.get("val", loaders["train"]), criterion, None, device, False)
        logger.info("[%02d/%d] train mse %.4f mae %.4f | val mse %.4f mae %.4f",
                    ep + 1, epochs, tr[0], tr[1], va[0], va[1])
        bundle = {"model_state_dict": model.state_dict(), "backbone": cfg.get("backbone", "resnet50"),
                  "img_size": img_size, "task": "regression", "target": "power_loss"}
        save_checkpoint(bundle, out_dir / "last.pth")
        if va[0] < best:
            best, wait = va[0], 0
            save_checkpoint(bundle, out_dir / "model.pth")
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", ep + 1)
                break

    results = {"best_val_mse": best}
    if "test" in loaders:
        te = _run_epoch(model, loaders["test"], criterion, None, device, False)
        results.update({"test_mse": te[0], "test_mae": te[1], "test_rmse": te[2]})
        logger.info("TEST  mse %.4f  mae %.4f  rmse %.4f", te[0], te[1], te[2])
    save_json(results, out_dir / "metrics_test.json")
    return results


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="DeepSolarEye power-loss regression (severity Track B).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("manifest", help="Build a regression manifest by parsing filenames.")
    pm.add_argument("--data-root", required=True)
    pm.add_argument("--out", required=True)
    pm.add_argument("--regex", default=DEFAULT_LABEL_REGEX)
    pm.add_argument("--repo-root", default=None)

    pt = sub.add_parser("train", help="Train the regressor from a config's track_b block.")
    pt.add_argument("--config", default="configs/severity.yaml")
    pt.add_argument("--manifest", default=None)
    pt.add_argument("--out-dir", default=None)
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--num-workers", type=int, default=None)
    pt.add_argument("--batch-size", type=int, default=None)
    pt.add_argument("--backbone", default=None)
    pt.add_argument("--limit", type=int, default=None, help="Subsample N images (faster runs).")
    args = ap.parse_args(argv)

    if args.cmd == "manifest":
        build_regression_manifest(args.data_root, args.out, args.regex, repo_root=args.repo_root)
        return

    full = load_config(args.config)
    cfg = full.get("track_b", full)  # severity.yaml nests track_b
    cfg = dict(cfg)
    cfg["repo_root"] = str(Path.cwd())
    for key, val in (("manifest", args.manifest), ("out_dir", args.out_dir), ("epochs", args.epochs),
                     ("num_workers", args.num_workers), ("batch_size", args.batch_size),
                     ("backbone", args.backbone), ("limit", args.limit)):
        if val is not None:
            cfg[key] = val
    train_regressor(cfg)


if __name__ == "__main__":
    main()
