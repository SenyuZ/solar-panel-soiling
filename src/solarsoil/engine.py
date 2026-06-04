"""Reusable training / inference loops and checkpoint IO.

Kept backend-aware: CUDA uses AMP (fp16) like the original notebook; Apple MPS
and CPU run in fp32 (GradScaler is CUDA-only).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    use_amp: bool = False,
    max_steps: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One training pass. Returns (mean_loss, y_true, y_pred)."""
    model.train()
    running, ys, ps = 0.0, [], []
    seen = 0
    for i, (xb, yb) in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        running += loss.item() * xb.size(0)
        seen += xb.size(0)
        ps.extend(out.argmax(1).detach().cpu().numpy())
        ys.extend(yb.detach().cpu().numpy())

    return running / max(seen, 1), np.asarray(ys), np.asarray(ps)


@torch.no_grad()
def run_inference(model, loader, device, max_steps: int | None = None):
    """Evaluate ``model`` over ``loader``. Returns (y_true, y_pred, y_prob)."""
    model.eval()
    ys, ps, probs = [], [], []
    for i, (xb, yb) in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        p = out.softmax(1)
        ps.extend(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        ys.extend(yb.numpy())
    return np.asarray(ys), np.asarray(ps), np.concatenate(probs) if probs else np.empty((0,))


def save_checkpoint(state: dict, path: str | Path) -> None:
    """Atomic checkpoint save (avoids half-written files on interruption)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location=None) -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
