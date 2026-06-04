"""Hand-crafted image-processing features for the classical baseline.

The deep model only *fine-tunes* a pretrained CNN — it demonstrates transfer
learning but not the image-processing fundamentals the course is about. This
module builds an interpretable feature vector from classic descriptors:

* **Colour** — per-channel HSV histograms + RGB colour moments (dust shifts hue
  toward grey/brown and lowers saturation).
* **Texture** — GLCM (Haralick) properties + uniform Local Binary Patterns
  (soiling and damage change surface texture).
* **Edges** — Canny edge density + Sobel gradient statistics (cracks add edges;
  uniform dust suppresses the panel's regular grid lines).

Feeding these to an SVM / Random Forest gives a transparent classical-vs-deep
comparison (Phase 2 benchmark).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _load(image, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb HxWx3 uint8, gray HxW uint8) resized to ``size``."""
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(image)).convert("RGB")
    img = img.resize((size, size))
    return np.asarray(img), np.asarray(img.convert("L"))


def _color_features(rgb: np.ndarray, bins: int = 16) -> tuple[list[float], list[str]]:
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    vals: list[float] = []
    names: list[str] = []
    for ch, nm, rng in [(0, "H", 180), (1, "S", 256), (2, "V", 256)]:
        h = cv2.calcHist([hsv], [ch], None, [bins], [0, rng]).flatten()
        h = h / (h.sum() + 1e-8)
        vals.extend(h.tolist())
        names.extend([f"hsv{nm}_{i}" for i in range(bins)])
    for c, nm in enumerate(["R", "G", "B"]):
        ch = rgb[:, :, c].astype(np.float32) / 255.0
        vals.extend([float(ch.mean()), float(ch.std())])
        names.extend([f"{nm}_mean", f"{nm}_std"])
    return vals, names


def _texture_features(gray: np.ndarray) -> tuple[list[float], list[str]]:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

    vals: list[float] = []
    names: list[str] = []
    # GLCM over 8 grey levels, averaged across 4 orientations.
    q = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(
        q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8, symmetric=True, normed=True,
    )
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        vals.append(float(graycoprops(glcm, prop).mean()))
        names.append(f"glcm_{prop}")
    # Uniform LBP histogram.
    P, R = 8, 1
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    vals.extend(hist.tolist())
    names.extend([f"lbp_{i}" for i in range(n_bins)])
    return vals, names


def _edge_features(gray: np.ndarray) -> tuple[list[float], list[str]]:
    import cv2

    edges = cv2.Canny(gray, 100, 200)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    vals = [float((edges > 0).mean()), float(mag.mean()), float(mag.std())]
    names = ["edge_density", "grad_mean", "grad_std"]
    return vals, names


def extract_features(image, size: int = 256) -> np.ndarray:
    """Return the concatenated feature vector for one image."""
    rgb, gray = _load(image, size)
    vals: list[float] = []
    for fn in (_color_features, _edge_features):
        v, _ = fn(rgb if fn is _color_features else gray)
        vals.extend(v)
    tv, _ = _texture_features(gray)
    vals.extend(tv)
    return np.asarray(vals, dtype=np.float32)


def feature_names(size: int = 256) -> list[str]:
    """Names aligned with :func:`extract_features` output (for RF importances)."""
    rgb, gray = _load(np.zeros((size, size, 3), dtype=np.uint8), size)
    names: list[str] = []
    _, cn = _color_features(rgb)
    _, en = _edge_features(gray)
    _, tn = _texture_features(gray)
    names.extend(cn)
    names.extend(en)
    names.extend(tn)
    return names


def extract_dataset(
    manifest: str | Path,
    split: str,
    repo_root: str | Path | None = None,
    size: int = 256,
    limit: int | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for every image in a manifest split.

    Returns ``(X, y)`` with ``X`` shape (n, d) and ``y`` the string labels.
    """
    import pandas as pd

    repo_root = Path(repo_root) if repo_root else Path.cwd()
    df = pd.read_csv(manifest)
    df = df[df["split"] == split].reset_index(drop=True)
    if limit:
        # Shuffle before truncating so a small smoke sample still spans classes
        # (the manifest is grouped by class on disk).
        df = df.sample(n=min(limit, len(df)), random_state=0).reset_index(drop=True)

    rows = df.itertuples(index=False)
    if show_progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=len(df), desc=f"features[{split}]")

    X, y = [], []
    for row in rows:
        X.append(extract_features(repo_root / row.filepath, size=size))
        y.append(row.label)
    return np.vstack(X), np.asarray(y)
