"""Train a CNN classifier (config-driven) — the modular replacement for the
notebook's training cells.

Two-stage transfer learning, faithful to the original:
1. freeze the backbone, train only the new head (``lr_head``);
2. after ``unfreeze_after`` epochs, unfreeze everything and fine-tune at a lower
   rate (``lr_finetune``).

Adds: stable seeds, MPS/CPU support, a committed manifest split, per-epoch metric
history (JSON+CSV), best/last checkpoints, early stopping, and a portable final
model bundle (weights + class names + backbone + img size) for evaluate/predict.

Example::

    python -m solarsoil.train --config configs/binary.yaml
    python -m solarsoil.train --config configs/binary.yaml --epochs 1 --max-steps 5  # smoke
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from . import metrics as M
from .data.datasets import make_dataloaders
from .data.manifest import build_manifest
from .engine import amp_enabled, load_checkpoint, run_inference, save_checkpoint, train_one_epoch
from .models.cnn import build_model, set_trainable, split_parameters
from .utils import count_parameters, get_device, load_config, save_json, set_seed, setup_logging

logger = logging.getLogger("solarsoil.train")


def _build_optimizer(params, lr: float, weight_decay: float):
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def train(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 123)))
    device = get_device(cfg.get("device", "auto"))
    repo_root = Path(cfg.get("repo_root", Path.cwd()))
    out_dir = Path(cfg.get("out_dir", "artifacts/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    task = cfg.get("task", "binary")
    logger.info("Device: %s | task: %s | out_dir: %s", device, task, out_dir)

    # Ensure the manifest exists (build it from data_root if missing).
    manifest = Path(cfg["manifest"])
    if not manifest.exists():
        logger.info("Manifest %s missing — building from %s", manifest, cfg["data_root"])
        build_manifest(
            data_root=cfg["data_root"],
            out_csv=manifest,
            label_space=task,
            seed=int(cfg.get("seed", 123)),
            repo_root=repo_root,
        )

    data = make_dataloaders(
        manifest=manifest,
        label_space=task,
        img_size=int(cfg.get("img_size", 224)),
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 4)),
        augment=bool(cfg.get("augment", True)),
        repo_root=repo_root,
        pin_memory=(device.type == "cuda"),
    )
    class_names = data["class_names"]
    logger.info("Classes %s | train counts %s", class_names, dict(zip(class_names, data["class_counts"])))

    model, head = build_model(cfg.get("backbone", "resnet50"), len(class_names), cfg.get("pretrained", True))
    model.to(device)
    head_params, backbone_params = split_parameters(model, head)

    weight = data["class_weights"].to(device) if cfg.get("use_class_weights", True) else None
    criterion = nn.CrossEntropyLoss(weight=weight)

    # ---- Stage 1: feature extraction (frozen backbone) ----
    set_trainable(backbone_params, False)
    set_trainable(head_params, True)
    optimizer = _build_optimizer(head_params, float(cfg.get("lr_head", 1e-4)), float(cfg.get("weight_decay", 0.0)))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.get("step_size", 7)), gamma=float(cfg.get("gamma", 0.1)))
    logger.info("Params trainable/total: %s/%s", *count_parameters(model))

    use_amp = amp_enabled(device)
    scaler = torch.amp.GradScaler() if use_amp else None

    epochs = int(cfg.get("epochs", 20))
    unfreeze_after = int(cfg.get("unfreeze_after", 5))
    patience = int(cfg.get("patience", 5))
    max_steps = cfg.get("max_steps")

    history: list[dict] = []
    best_val = float("inf")
    wait = 0
    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"

    for epoch in range(epochs):
        if epoch == unfreeze_after and unfreeze_after < epochs:
            logger.info("Unfreezing backbone for full fine-tuning")
            set_trainable(backbone_params, True)
            optimizer = _build_optimizer(model.parameters(), float(cfg.get("lr_finetune", 5e-5)), float(cfg.get("weight_decay", 0.0)))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.get("step_size", 7)), gamma=float(cfg.get("gamma", 0.1)))

        tr_loss, tr_y, tr_p = train_one_epoch(
            model, data["loaders"]["train"], criterion, optimizer, device,
            scaler=scaler, use_amp=use_amp, max_steps=max_steps,
        )
        tr_m = M.compute_metrics(tr_y, tr_p, class_names)

        val_loader = data["loaders"].get("val", data["loaders"]["train"])
        va_y, va_p, _ = run_inference(model, val_loader, device, max_steps=max_steps)
        va_m = M.compute_metrics(va_y, va_p, class_names)
        # val loss for early stopping
        with torch.no_grad():
            va_loss = _val_loss(model, val_loader, criterion, device, max_steps)
        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "train_loss": tr_loss, "train_acc": tr_m["accuracy"], "train_f1": tr_m["f1"],
            "val_loss": va_loss, "val_acc": va_m["accuracy"], "val_f1": va_m["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        logger.info(
            "[%02d/%d] train L %.4f A %.3f F1 %.3f | val L %.4f A %.3f F1 %.3f",
            epoch + 1, epochs, tr_loss, tr_m["accuracy"], tr_m["f1"], va_loss, va_m["accuracy"], va_m["f1"],
        )

        bundle = {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "backbone": cfg.get("backbone", "resnet50"),
            "img_size": int(cfg.get("img_size", 224)),
            "task": task,
            "epoch": epoch + 1,
        }
        save_checkpoint(bundle, last_path)
        if va_loss < best_val:
            best_val, wait = va_loss, 0
            save_checkpoint(bundle, best_path)
            logger.info("  ✓ new best (val_loss=%.4f)", best_val)
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Persist final (best) bundle + training history.
    best = load_checkpoint(best_path, map_location=device)
    save_checkpoint(best, out_dir / "model.pth")
    save_json({"config": {k: v for k, v in cfg.items() if not k.startswith("_")},
               "best_val_loss": best_val, "history": history}, out_dir / "history.json")
    _write_history_csv(history, out_dir / "history.csv")
    logger.info("Done. Best val loss %.4f. Final model: %s", best_val, out_dir / "model.pth")
    return {"best_val_loss": best_val, "history": history, "out_dir": str(out_dir)}


@torch.no_grad()
def _val_loss(model, loader, criterion, device, max_steps=None) -> float:
    model.eval()
    total, seen = 0.0, 0
    for i, (xb, yb) in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        xb, yb = xb.to(device), yb.to(device)
        total += criterion(model(xb), yb).item() * xb.size(0)
        seen += xb.size(0)
    return total / max(seen, 1)


def _write_history_csv(history: list[dict], path: Path) -> None:
    if not history:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Train a solar-panel CNN classifier.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="Cap batches/epoch (smoke test).")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["repo_root"] = str(Path.cwd())
    for key, val in (("epochs", args.epochs), ("max_steps", args.max_steps),
                     ("batch_size", args.batch_size), ("num_workers", args.num_workers),
                     ("device", args.device), ("out_dir", args.out_dir)):
        if val is not None:
            cfg[key] = val
    train(cfg)


if __name__ == "__main__":
    main()
