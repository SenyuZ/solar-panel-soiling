"""Run a trained model on one image or a folder, optionally with a Grad-CAM overlay.

Example::

    python -m solarsoil.predict --model artifacts/binary/model.pth --image path/to/panel.jpg
    python -m solarsoil.predict --model artifacts/binary/model.pth --image folder/ --gradcam
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image

from .data.datasets import build_transforms
from .data.dedup import IMAGE_EXTS
from .engine import load_checkpoint
from .models.cnn import build_model
from .utils import get_device, setup_logging

logger = logging.getLogger("solarsoil.predict")


def _gather_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return [path]


def load_model(model_path: str | Path, device):
    bundle = load_checkpoint(model_path, map_location=device)
    model, _ = build_model(bundle["backbone"], len(bundle["class_names"]), pretrained=False)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device).eval()
    return model, bundle


@torch.no_grad()
def predict_image(model, bundle, image_path: Path, device) -> dict:
    tfm = build_transforms(bundle["img_size"], train=False)
    with Image.open(image_path) as img:
        x = tfm(img).unsqueeze(0).to(device)
    probs = model(x).softmax(1).cpu().numpy()[0]
    idx = int(probs.argmax())
    class_names = bundle["class_names"]
    return {
        "image": str(image_path),
        "prediction": class_names[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(class_names, probs)},
    }


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Predict solar-panel condition for image(s).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True, help="Image file or directory.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gradcam", action="store_true", help="Also save a Grad-CAM overlay.")
    ap.add_argument("--out-dir", default="reports/figures/gradcam")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    model, bundle = load_model(args.model, device)

    images = _gather_images(Path(args.image))
    for img_path in images:
        res = predict_image(model, bundle, img_path, device)
        probs = "  ".join(f"{c}={p:.3f}" for c, p in res["probabilities"].items())
        print(f"{img_path.name:40s} -> {res['prediction']:16s} ({res['confidence']:.3f})  [{probs}]")

        if args.gradcam:
            from .explain.gradcam import save_gradcam_overlay  # lazy import

            out = Path(args.out_dir) / f"{img_path.stem}_gradcam.png"
            save_gradcam_overlay(model, bundle, img_path, out, device=device)
            logger.info("Saved Grad-CAM overlay: %s", out)


if __name__ == "__main__":
    main()
