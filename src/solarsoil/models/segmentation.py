"""Soiling segmentation (U-Net) → measured per-image dirt coverage.

Trains a compact, from-scratch **U-Net** on the dust-segmentation dataset
(image -> binary dust mask, 0=background / 1=dust) to predict *which pixels are
soiled*. This upgrades the classical, approximate Track-A soiling index
(:mod:`solarsoil.severity`) into a **measured coverage percentage** plus a
pixel-level localisation of the dirt.

Run::

    python -m solarsoil.data.download --source dust_seg
    python -m solarsoil.models.segmentation manifest \
        --images Data/raw/dust_seg/images --masks Data/raw/dust_seg/masks \
        --out manifests/segmentation_manifest.csv
    python -m solarsoil.models.segmentation train --config configs/segmentation.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ..data.dedup import IMAGE_EXTS
from ..engine import load_checkpoint, save_checkpoint
from ..utils import get_device, load_config, save_json, set_seed, setup_logging

logger = logging.getLogger("solarsoil.segmentation")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --------------------------- U-Net ---------------------------
class _DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Standard 4-level U-Net. Input side must be a multiple of 16."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32) -> None:
        super().__init__()
        self.d1, self.d2 = _DoubleConv(in_ch, base), _DoubleConv(base, base * 2)
        self.d3, self.d4 = _DoubleConv(base * 2, base * 4), _DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.u4 = _DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = _DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = _DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        b = self.bottleneck(self.pool(c4))
        x = self.u4(torch.cat([self.up4(b), c4], 1))
        x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.out(x)  # logits (B, 1, H, W)


# --------------------------- data ---------------------------
def _to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def build_seg_manifest(images_dir, masks_dir, out_csv, val_size=0.1, test_size=0.1,
                       seed=123, repo_root=None) -> pd.DataFrame:
    """Pair images with their masks and write a split manifest CSV."""
    repo_root = Path(repo_root) if repo_root else Path.cwd()
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    rows = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS or "checkpoint" in img.name.lower():
            continue
        mask = masks_dir / (img.stem + ".png")
        if not mask.exists():
            continue
        rows.append({
            "image": img.resolve().relative_to(repo_root.resolve()).as_posix(),
            "mask": mask.resolve().relative_to(repo_root.resolve()).as_posix(),
        })
    if not rows:
        raise RuntimeError(f"No image/mask pairs under {images_dir} / {masks_dir}")
    df = pd.DataFrame(rows).reset_index(drop=True)
    tr, tmp = train_test_split(df.index, test_size=val_size + test_size, random_state=seed)
    va, te = train_test_split(tmp, test_size=test_size / (val_size + test_size), random_state=seed)
    df["split"] = "train"
    df.loc[va, "split"], df.loc[te, "split"] = "val", "test"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d pairs): %s", out_csv, len(df), df["split"].value_counts().to_dict())
    return df


class SegDataset(Dataset):
    def __init__(self, manifest, split, img_size=256, train=False, repo_root=None):
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.size, self.train = img_size, train
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = _to_tensor(Image.open(self.repo_root / r["image"]), self.size)
        mask = Image.open(self.repo_root / r["mask"]).convert("L").resize((self.size, self.size), Image.NEAREST)
        m = torch.from_numpy((np.asarray(mask) > 127).astype(np.float32))[None]
        if self.train and torch.rand(1).item() < 0.5:  # horizontal flip
            x, m = torch.flip(x, [2]), torch.flip(m, [2])
        return x, m


# --------------------------- loss / metrics ---------------------------
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return (1 - num / den).mean()


def seg_scores(logits, target, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum((1, 2, 3))
    dice = (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    iou = (inter + 1) / (((p + target) > 0).float().sum((1, 2, 3)) + 1)
    return dice.mean().item(), iou.mean().item()


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    ds, iu = [], []
    for x, m in loader:
        d, i = seg_scores(model(x.to(device)), m.to(device))
        ds.append(d)
        iu.append(i)
    return float(np.mean(ds)), float(np.mean(iu))


# --------------------------- train ---------------------------
def train_segmenter(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 123)))
    device = get_device(cfg.get("device", "auto"))
    repo_root = Path(cfg.get("repo_root", Path.cwd()))
    out_dir = Path(cfg.get("out_dir", "artifacts/segmentation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    img_size = int(cfg.get("img_size", 256))
    base = int(cfg.get("base_channels", 32))

    df = pd.read_csv(cfg["manifest"])
    if cfg.get("limit"):
        frac = min(1.0, int(cfg["limit"]) / len(df))
        df = pd.concat([g.sample(frac=frac, random_state=123) for _, g in df.groupby("split")]).reset_index(drop=True)

    loaders = {}
    for sp in ("train", "val", "test"):
        if sp not in set(df["split"]):
            continue
        ds = SegDataset(df, sp, img_size, train=(sp == "train"), repo_root=repo_root)
        loaders[sp] = DataLoader(ds, batch_size=int(cfg.get("batch_size", 16)),
                                 shuffle=(sp == "train"), num_workers=int(cfg.get("num_workers", 2)))

    model = UNet(base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    bce = nn.BCEWithLogitsLoss()
    epochs, patience = int(cfg.get("epochs", 15)), int(cfg.get("patience", 5))
    best, wait = -1.0, 0
    max_steps = cfg.get("max_steps")

    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for step, (x, m) in enumerate(loaders["train"]):
            if max_steps is not None and step >= max_steps:
                break
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            out = model(x)
            loss = bce(out, m) + dice_loss(out, m)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        vd, vi = _eval(model, loaders.get("val", loaders["train"]), device)
        logger.info("[%02d/%d] train loss %.4f | val dice %.4f iou %.4f", ep + 1, epochs, tot / max(n, 1), vd, vi)
        bundle = {"model_state_dict": model.state_dict(), "img_size": img_size,
                  "base_channels": base, "task": "segmentation"}
        save_checkpoint(bundle, out_dir / "last.pth")
        if vd > best:
            best, wait = vd, 0
            save_checkpoint(bundle, out_dir / "model.pth")
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", ep + 1)
                break

    res = {"best_val_dice": best}
    if "test" in loaders:
        td, ti = _eval(model, loaders["test"], device)
        res.update({"test_dice": td, "test_iou": ti})
        logger.info("TEST  dice %.4f  iou %.4f", td, ti)
    save_json(res, out_dir / "metrics_test.json")
    return res


# --------------------------- inference / coverage ---------------------------
def load_segmenter(model_path, device):
    bundle = load_checkpoint(model_path, map_location=device)
    model = UNet(base=bundle.get("base_channels", 32))
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device).eval()
    return model, bundle


@torch.no_grad()
def predict_dust(model, bundle, image, device):
    """Return a dust-probability map in [0,1] at the model's working resolution."""
    img = image if isinstance(image, Image.Image) else Image.open(image)
    x = _to_tensor(img, bundle["img_size"]).unsqueeze(0).to(device)
    return torch.sigmoid(model(x))[0, 0].cpu().numpy()


def coverage_from_prob(prob, thr: float = 0.5) -> float:
    """Measured dust coverage fraction = soiled pixels / total pixels."""
    return float((prob > thr).mean())


def overlay_dust(image, prob, thr: float = 0.5, alpha: float = 0.45) -> Image.Image:
    """Tint predicted dust pixels red over the image, at the image's native size.

    The probability map (predicted at the model's square working size) is resized
    back to the original aspect ratio so the overlay isn't distorted.
    """
    base = (image if isinstance(image, Image.Image) else Image.open(image)).convert("RGB")
    w, h = base.size
    prob_full = np.asarray(
        Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0
    base_np = np.asarray(base, dtype=np.float32) / 255.0
    red = np.zeros_like(base_np)
    red[..., 0] = 1.0
    mask = (prob_full > thr)[..., None]
    blend = np.where(mask, (1 - alpha) * base_np + alpha * red, base_np)
    return Image.fromarray((blend * 255).clip(0, 255).astype(np.uint8))


def save_prediction(model, bundle, image_path, out_path, device, thr: float = 0.5) -> dict:
    """Predict the dust mask for one image; save an overlay; return coverage %."""
    prob = predict_dust(model, bundle, image_path, device)
    overlay = overlay_dust(Image.open(image_path), prob, thr)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)
    return {"coverage_pct": round(100 * coverage_from_prob(prob, thr), 1), "overlay": str(out_path)}


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="U-Net soiling segmentation (measured coverage).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("manifest", help="Pair images with masks into a split manifest.")
    pm.add_argument("--images", required=True)
    pm.add_argument("--masks", required=True)
    pm.add_argument("--out", required=True)
    pm.add_argument("--repo-root", default=None)
    pt = sub.add_parser("train", help="Train the U-Net from a config.")
    pt.add_argument("--config", default="configs/segmentation.yaml")
    pt.add_argument("--manifest", default=None)
    pt.add_argument("--out-dir", default=None)
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--limit", type=int, default=None)
    pt.add_argument("--max-steps", type=int, default=None)
    pt.add_argument("--num-workers", type=int, default=None)
    pp = sub.add_parser("predict", help="Predict dust mask + coverage % for an image/folder.")
    pp.add_argument("--model", required=True)
    pp.add_argument("--image", required=True)
    pp.add_argument("--out-dir", default="reports/figures/segmentation")
    pp.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args(argv)

    if args.cmd == "manifest":
        build_seg_manifest(args.images, args.masks, args.out, repo_root=args.repo_root)
        return
    if args.cmd == "predict":
        device = get_device("auto")
        model, bundle = load_segmenter(args.model, device)
        path = Path(args.image)
        images = ([p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
                  if path.is_dir() else [path])
        for img in images:
            out = Path(args.out_dir) / f"{img.stem}_dustseg.png"
            res = save_prediction(model, bundle, img, out, device, args.thr)
            print(f"{img.name:40s} coverage={res['coverage_pct']:5.1f}%  -> {res['overlay']}")
        return
    cfg = dict(load_config(args.config))
    cfg["repo_root"] = str(Path.cwd())
    for k, v in (("manifest", args.manifest), ("out_dir", args.out_dir), ("epochs", args.epochs),
                 ("limit", args.limit), ("max_steps", args.max_steps), ("num_workers", args.num_workers)):
        if v is not None:
            cfg[k] = v
    train_segmenter(cfg)


if __name__ == "__main__":
    main()
