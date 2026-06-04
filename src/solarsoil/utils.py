"""Shared utilities: reproducibility, device selection, config & IO helpers.

Kept dependency-light. ``torch`` is imported lazily inside the functions that
need it so the classical pipeline can run without a deep-learning stack.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger("solarsoil")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple, readable console logger (idempotent)."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int = 123, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if available) torch for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def get_device(prefer: str = "auto"):
    """Return the best available torch device.

    ``prefer`` may be ``"auto" | "cuda" | "mps" | "cpu"``. On Apple Silicon the
    Metal (``mps``) backend is selected automatically — the original Colab
    notebook only handled cuda/cpu.
    """
    import torch

    if prefer not in ("auto", "cuda", "mps", "cpu"):
        prefer = "auto"
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if prefer in ("auto", "mps") and mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    return cfg


def save_json(obj: Any, path: str | os.PathLike) -> None:
    """Write ``obj`` as pretty JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (a directory) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model) -> tuple[int, int]:
    """Return (trainable, total) parameter counts for a torch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
