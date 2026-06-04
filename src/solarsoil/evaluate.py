"""Evaluate a trained model bundle on a manifest split.

Reconstructs the model from the portable bundle saved by :mod:`train` (weights +
class names + backbone + img size), runs inference on a split (default ``test``),
prints headline + per-class metrics and writes metrics JSON + confusion-matrix
figures.

Example::

    python -m solarsoil.evaluate --model artifacts/binary/model.pth \
        --manifest manifests/binary_manifest.csv --split test
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from . import metrics as M
from .data.datasets import make_dataloaders
from .engine import load_checkpoint, run_inference
from .models.cnn import build_model
from .utils import get_device, save_json, setup_logging

logger = logging.getLogger("solarsoil.evaluate")


def evaluate_bundle(
    model_path: str | Path,
    manifest: str | Path,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
    device_pref: str = "auto",
    out_dir: str | Path | None = None,
    max_steps: int | None = None,
) -> dict:
    device = get_device(device_pref)
    bundle = load_checkpoint(model_path, map_location=device)
    class_names = bundle["class_names"]

    model, _ = build_model(bundle["backbone"], len(class_names), pretrained=False)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)

    data = make_dataloaders(
        manifest=manifest,
        label_space=bundle["task"],
        img_size=bundle["img_size"],
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        pin_memory=(device.type == "cuda"),
    )
    if split not in data["loaders"]:
        raise ValueError(f"Split {split!r} not in manifest (have {list(data['loaders'])})")

    y_true, y_pred, _ = run_inference(model, data["loaders"][split], device, max_steps=max_steps)
    results = M.compute_metrics(y_true, y_pred, class_names)
    print(M.format_metrics(results, title=f"{bundle['task']} / {split}"))

    out_dir = Path(out_dir) if out_dir else Path(model_path).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, out_dir / f"metrics_{split}.json")
    M.plot_confusion_matrix(y_true, y_pred, class_names, out_dir / f"confusion_{split}.png")
    M.plot_confusion_matrix(
        y_true, y_pred, class_names, out_dir / f"confusion_{split}_norm.png", normalize=True
    )
    logger.info("Wrote metrics + confusion matrices to %s", out_dir)
    return results


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Evaluate a trained model on a split.")
    ap.add_argument("--model", required=True, help="Path to model.pth / best.pth bundle.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args(argv)

    evaluate_bundle(
        model_path=args.model,
        manifest=args.manifest,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_pref=args.device,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
