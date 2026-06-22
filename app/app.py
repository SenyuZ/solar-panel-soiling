"""Interactive Gradio demo: drop a solar-panel photo, get condition + explanation.

Shows the whole pipeline in one place:
* CNN classification (binary or multi-class, whichever model is loaded) with
  class probabilities,
* a Grad-CAM heatmap of where the model looked,
* a classical soiling-coverage overlay with a soiling index + severity bucket, and
  a power-loss estimate — the measured DeepSolarEye Track B regressor when its
  checkpoint is present (``artifacts/severity/model.pth``), otherwise the classical
  illustrative heuristic.

Run::

    python app/app.py --model artifacts/binary/model.pth      # or condition/multiclass
    python app/app.py            # auto-detects a model under artifacts/, else
                                 # runs the classical soiling view only

If no trained model is found, classification/Grad-CAM are disabled and only the
(label-free) classical soiling estimate is shown — so the demo still works before
any training.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Gradio 6 enables server-side rendering (a Node proxy in front of Python) by default,
# which makes the app hang at "Starting" on Hugging Face Spaces (HF health-checks port
# 7860 but SSR puts the Python server on 7861). Disable it before gradio is imported so
# the Gradio server binds 7860 directly and HF detects it as Running.
os.environ.setdefault("GRADIO_SSR_MODE", "false")

import gradio as gr
from PIL import Image

from solarsoil.engine import load_checkpoint
from solarsoil.explain.gradcam import compute_cam, overlay_cam
from solarsoil.models.cnn import build_model
from solarsoil.severity import estimate_soiling, overlay_mask
from solarsoil.utils import get_device

# Populated by load_model(); kept module-global so the UI callback can use it.
STATE: dict = {"model": None, "bundle": None, "regressor": None,
               "reg_bundle": None, "segmenter": None, "seg_bundle": None,
               "device": get_device(), "_model_path": None,
               "_loaded": {"model": False, "regressor": False, "segmenter": False}}

# Classifier auto-detect. On Hugging Face Spaces, nested upload paths are fiddly, so the
# easiest overrides come first: a single `model.pth` at the app root, or the
# SOLARSOIL_MODEL env var / Space variable pointing anywhere.
_CANDIDATE_MODELS = [
    "model.pth",                        # drop-in override at the app root (easy on HF Spaces)
    "artifacts/condition/model.pth",
    "artifacts/multiclass/model.pth",
    "artifacts/binary/model.pth",
]

# Track B: measured power-loss regressor (DeepSolarEye). Optional — falls back to
# the classical illustrative estimate when this checkpoint is absent.
_SEVERITY_MODEL = "artifacts/severity/model.pth"

# ML dust segmentation (ResNet-34 U-Net). Optional — when present, the demo shows
# its learned dust mask next to the classical soiling overlay for comparison.
_SEG_MODEL = "artifacts/segmentation/model.pth"


def load_model(model_path: str | None) -> str:
    """Load a model bundle into STATE. Returns a human-readable status string."""
    if model_path is None:
        model_path = os.environ.get("SOLARSOIL_MODEL") or next(
            (p for p in _CANDIDATE_MODELS if Path(p).exists()), None)
    if model_path is None or not Path(model_path).exists():
        STATE["model"], STATE["bundle"] = None, None
        return "No trained model found — showing the classical soiling view only."

    device = STATE["device"]
    bundle = load_checkpoint(model_path, map_location=device)
    model, _ = build_model(bundle["backbone"], len(bundle["class_names"]), pretrained=False)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device).eval()
    STATE["model"], STATE["bundle"] = model, bundle
    return f"Loaded {bundle['task']} model ({bundle['backbone']}): {model_path}"


def load_regressor_into_state() -> None:
    """Load the Track B power-loss regressor into STATE if its checkpoint exists."""
    if not Path(_SEVERITY_MODEL).exists():
        STATE["regressor"], STATE["reg_bundle"] = None, None
        return
    try:
        from solarsoil.models.regression import load_regressor

        STATE["regressor"], STATE["reg_bundle"] = load_regressor(_SEVERITY_MODEL, STATE["device"])
    except Exception:  # noqa: BLE001 — demo stays usable on the classical fallback
        STATE["regressor"], STATE["reg_bundle"] = None, None


def load_segmenter_into_state() -> None:
    """Load the U-Net dust segmenter into STATE if its checkpoint exists."""
    if not Path(_SEG_MODEL).exists():
        STATE["segmenter"], STATE["seg_bundle"] = None, None
        return
    try:
        from solarsoil.models.segmentation import load_segmenter

        STATE["segmenter"], STATE["seg_bundle"] = load_segmenter(_SEG_MODEL, STATE["device"])
    except Exception:  # noqa: BLE001 — classical overlay still shown if this fails
        STATE["segmenter"], STATE["seg_bundle"] = None, None


def ensure_models() -> None:
    """Load models on first use and cache them, so the UI starts instantly instead
    of loading ~226 MB of checkpoints before the page is even served."""
    if not STATE["_loaded"]["model"]:
        load_model(STATE.get("_model_path"))
        STATE["_loaded"]["model"] = True
    if not STATE["_loaded"]["regressor"]:
        load_regressor_into_state()
        STATE["_loaded"]["regressor"] = True
    if not STATE["_loaded"]["segmenter"]:
        load_segmenter_into_state()
        STATE["_loaded"]["segmenter"] = True


def analyze(image: Image.Image):
    """Run the full pipeline on one PIL image for the Gradio callback."""
    if image is None:
        return {}, None, None, None, "Upload a solar-panel image to begin."

    ensure_models()  # lazy: load checkpoints on first use, then cached

    # --- Classical soiling estimate (always available) ---
    soil = estimate_soiling(image)
    soil_overlay = overlay_mask(image, soil["mask"])

    # --- ML dust segmentation (U-Net), if its checkpoint is present ---
    seg_overlay, seg_line = None, ""
    seg = STATE.get("segmenter")
    if seg is not None:
        from solarsoil.models.segmentation import (
            coverage_from_prob,
            overlay_dust,
            predict_dust,
        )

        prob = predict_dust(seg, STATE["seg_bundle"], image, STATE["device"])
        seg_overlay = overlay_dust(image, prob)
        seg_line = (
            "_U-Net dust overlay (exploratory): trained on inconsistent third-party "
            "labels, so it is shown for visual comparison only, not a validated "
            "measurement (see the repo's limitations)._"
        )

    # --- Power loss: prefer the measured Track B regressor, else classical ---
    reg = STATE.get("regressor")
    if reg is not None:
        from solarsoil.models.regression import predict_power_loss

        pl = predict_power_loss(reg, STATE["reg_bundle"], image, STATE["device"]) * 100
        power_line = (
            f"**power loss ≈ {pl:.1f}%** _(DeepSolarEye regressor, test MAE 0.075; "
            f"accurate on panel-filling photos — cross-domain & indicative on wide, "
            f"background-heavy shots)_"
        )
    else:
        power_line = (
            f"**est. power loss ≈ {soil['power_loss_pct']:.1f}%** "
            f"_(classical heuristic — train Track B for a measured estimate)_"
        )

    _parts = [
        f"**Soiling index:** {soil['soiling_index']:.3f}  ·  "
        f"**severity:** {soil['severity']}  ·  "
        f"**coverage ≈** {soil['coverage_pct']:.1f}% _(classical)_",
        power_line,
    ]
    if seg_line:
        _parts.append(seg_line)
    _parts.append(
        "_Soiling index is an unsupervised relative measure (desaturation vs a "
        "clean reference); see DATASET.md / README for its limits._"
    )
    summary = "\n\n".join(_parts)

    model, bundle = STATE["model"], STATE["bundle"]
    if model is None:
        return ({}, None, soil_overlay, seg_overlay,
                "**No trained model loaded.**\n\n" + summary)

    device = STATE["device"]
    cam, idx, _ = compute_cam(model, bundle, image, device)
    probs = {c: float(p) for c, p in _class_probs(model, bundle, image, device).items()}
    cam_overlay = overlay_cam(image, cam, bundle["img_size"])
    head = f"**Prediction:** {bundle['class_names'][idx]}  ({bundle['task']} model)\n\n"
    return probs, cam_overlay, soil_overlay, seg_overlay, head + summary


def _class_probs(model, bundle, image, device) -> dict:
    import torch

    from solarsoil.data.datasets import build_transforms

    tfm = build_transforms(bundle["img_size"], train=False)
    with torch.no_grad():
        x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
        p = model(x).softmax(1).cpu().numpy()[0]
    return {c: float(v) for c, v in zip(bundle["class_names"], p)}


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Solar Panel Soiling Analyzer") as demo:
        gr.Markdown(
            "# Solar Panel Soiling Analyzer\n"
            "Upload a panel photo to get its **condition** (CNN) with a **Grad-CAM** "
            "explanation, plus two takes on *where* the dirt is — a **classical** "
            "image-processing estimate and a **U-Net** ML segmentation — side by side."
        )
        gr.Markdown(
            "_Models load on the first analysis (a few seconds on the free CPU "
            "tier), then it's fast._"
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Solar panel image")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                label = gr.Label(label="Predicted condition (probabilities)")
                summary = gr.Markdown()
        with gr.Row():
            cam_img = gr.Image(label="Grad-CAM (where the classifier looked)")
        with gr.Row():
            soil_img = gr.Image(label="Soiling coverage — classical (no ML)")
            seg_img = gr.Image(label="Dust segmentation — U-Net (ML)")
        outputs = [label, cam_img, soil_img, seg_img, summary]
        btn.click(analyze, inputs=inp, outputs=outputs)
        inp.change(analyze, inputs=inp, outputs=outputs)
    return demo


# Hugging Face Spaces imports this file and serves the top-level `demo`. Building it
# is cheap now (models load lazily on first use), so the app reaches "Running" fast
# instead of waiting on ~226 MB of checkpoints. Exposing `demo` also avoids HF's
# fallback launch path (the source of the asyncio "Invalid file descriptor" noise).
demo = build_demo()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Launch the solar-panel soiling demo.")
    ap.add_argument("--model", default=None, help="Path to a model.pth bundle.")
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    # The model path is read lazily on the first analysis, so just record it and launch.
    if args.model:
        STATE["_model_path"] = args.model
    demo.launch(share=args.share, server_port=args.port, ssr_mode=False)


if __name__ == "__main__":
    main()
