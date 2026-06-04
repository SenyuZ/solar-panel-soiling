"""Soiling severity / coverage quantification.

Two complementary tracks, matching the plan:

* **Track A (here, runs now):** an *unsupervised, classical image-processing*
  soiling estimate. On this dataset the only low-level cue that reliably
  separates clean from dusty panels is **saturation** — dust greys the surface
  out (clean mean S≈0.32 vs dusty≈0.22), while brightness and texture barely
  move (clean and dusty cells are both smooth). So Track A measures, per pixel,
  how desaturated the panel is relative to a clean-panel reference, and reports a
  continuous **soiling index** (the headline number), an approximate coverage,
  a severity bucket, and an illustrative power-loss estimate.

  Honest limitation: clean and dusty panels overlap a lot in low-level statistics
  (precisely why a CNN is worthwhile), so Track A is best read as a *relative*
  index and a *localisation* aid, not a calibrated coverage percentage.

* **Track B (data-grounded):** train a CNN regression head on DeepSolarEye (45k
  images labelled with measured % power loss) for a trustworthy quantitative
  estimate, with Grad-CAM (Phase 5) for weakly-supervised localisation. See
  ``train_power_loss_regressor`` and ``configs/severity.yaml``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .data.dedup import IMAGE_EXTS, iter_images
from .utils import setup_logging

logger = logging.getLogger("solarsoil.severity")

# Mean saturation of *clean* panels in the project dataset (calibration ref).
# Re-estimate for a new dataset with ``calibrate_reference``.
DEFAULT_S_REF = 0.32

# soiling-index -> severity bucket (calibrated to this dataset; clean≈0.32, dusty≈0.47).
SEVERITY_BINS = [(0.30, "clean"), (0.40, "light"), (0.55, "moderate"), (1.01, "heavy")]


def severity_bucket(soiling_index: float) -> str:
    for hi, name in SEVERITY_BINS:
        if soiling_index < hi:
            return name
    return "heavy"


def estimate_power_loss(soiling_index: float, k: float = 0.6) -> float:
    """Illustrative soiling-index -> power-loss fraction in [0, 1].

    Deliberately simple; the data-driven Track B regressor replaces this with a
    value learned from measured power loss.
    """
    return float(min(1.0, max(0.0, k * soiling_index)))


def _load_rgb(image, size: int = 512) -> np.ndarray:
    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = Image.fromarray(np.asarray(image)).convert("RGB")
    return np.asarray(img.resize((size, size)))


def soiling_score_map(rgb: np.ndarray, s_ref: float = DEFAULT_S_REF) -> np.ndarray:
    """Per-pixel soiling score in [0, 1]: desaturation relative to ``s_ref``.

    A pixel scores high when its saturation is well below the clean reference
    (dust washes colour out). Very dark pixels (frames, shadows, where hue is
    unreliable) are gated out by a brightness floor.
    """
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0
    desat = np.clip((s_ref - sat) / (s_ref + 1e-6), 0.0, 1.0)
    desat *= (val > 0.15)  # gate out near-black pixels
    return cv2.GaussianBlur(desat, (0, 0), 3)


def estimate_soiling(
    image,
    size: int = 512,
    s_ref: float = DEFAULT_S_REF,
    coverage_threshold: float = 0.5,
) -> dict:
    """Estimate soiling for one panel image (Track A).

    Returns ``soiling_index`` (mean desaturation, the primary number),
    ``coverage`` (fraction above ``coverage_threshold``, approximate),
    ``severity``, ``power_loss_pct`` (illustrative), and the binary ``mask``.
    """
    rgb = _load_rgb(image, size)
    score = soiling_score_map(rgb, s_ref)
    soiling_index = float(score.mean())
    mask = (score >= coverage_threshold).astype(np.uint8)
    coverage = float(mask.mean())
    return {
        "soiling_index": round(soiling_index, 3),
        "coverage": coverage,
        "coverage_pct": round(100 * coverage, 1),
        "severity": severity_bucket(soiling_index),
        "power_loss_pct": round(100 * estimate_power_loss(soiling_index), 1),
        "mask": mask,
    }


def overlay_mask(image, mask: np.ndarray, size: int = 512, alpha: float = 0.45) -> Image.Image:
    """Tint the soiled mask red over the original image (localisation view)."""
    rgb = _load_rgb(image, size).astype(np.float32) / 255.0
    red = np.zeros_like(rgb)
    red[:, :, 0] = 1.0
    m = mask.astype(bool)[..., None]
    blended = np.where(m, (1 - alpha) * rgb + alpha * red, rgb)
    return Image.fromarray((blended * 255).clip(0, 255).astype(np.uint8))


def save_soiling_overlay(image, out_path, size: int = 512, s_ref: float = DEFAULT_S_REF) -> dict:
    res = estimate_soiling(image, size=size, s_ref=s_ref)
    overlay = overlay_mask(image, res["mask"], size=size)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)
    res = {k: v for k, v in res.items() if k != "mask"}
    res["overlay"] = str(out_path)
    return res


def calibrate_reference(clean_images, size: int = 512, percentile: float = 50.0) -> float:
    """Estimate ``s_ref`` (clean saturation reference) from clean sample images."""
    import cv2

    sats = []
    for img in clean_images:
        rgb = _load_rgb(img, size)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        sats.append(np.percentile(hsv[:, :, 1] / 255.0, percentile))
    ref = float(np.mean(sats)) if sats else DEFAULT_S_REF
    logger.info("Calibrated s_ref=%.3f from %d clean images", ref, len(sats))
    return ref


def train_power_loss_regressor(*_args, **_kwargs):  # pragma: no cover - needs dataset
    """Track B entry point (requires the DeepSolarEye dataset).

    Reuses the CNN backbone with a single-output head trained with MSE against
    measured % power loss. Kept as an explicit, optional path because it depends
    on the large DeepSolarEye download.
    """
    raise NotImplementedError(
        "Track B needs the DeepSolarEye dataset. Download it with "
        "`python -m solarsoil.data.download --source deepsolareye`, build an "
        "(image -> power_loss) regression manifest, then train a 1-output CNN "
        "head with MSE. See configs/severity.yaml and the README roadmap."
    )


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Estimate soiling (Track A, classical).")
    ap.add_argument("--image", required=True, help="Image file or directory.")
    ap.add_argument("--out-dir", default="reports/figures/coverage")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--s-ref", type=float, default=DEFAULT_S_REF)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    path = Path(args.image)
    images = iter_images(path) if path.is_dir() else [path]
    if args.limit:
        images = images[: args.limit]
    for img in images:
        out = Path(args.out_dir) / f"{img.stem}_soiling.png"
        res = save_soiling_overlay(img, out, size=args.size, s_ref=args.s_ref)
        print(
            f"{img.name:40s} index={res['soiling_index']:.3f}  "
            f"severity={res['severity']:8s}  coverage≈{res['coverage_pct']:5.1f}%  "
            f"est_power_loss≈{res['power_loss_pct']:5.1f}%"
        )


if __name__ == "__main__":
    main()
