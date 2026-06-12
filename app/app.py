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
from pathlib import Path

import gradio as gr
from PIL import Image

from solarsoil.engine import load_checkpoint
from solarsoil.explain.gradcam import compute_cam, overlay_cam
from solarsoil.models.cnn import build_model
from solarsoil.severity import estimate_soiling, overlay_mask
from solarsoil.utils import get_device

# Populated by load_model(); kept module-global so the UI callback can use it.
STATE: dict = {"model": None, "bundle": None, "regressor": None,
               "reg_bundle": None, "device": get_device()}

_CANDIDATE_MODELS = [
    "artifacts/condition/model.pth",
    "artifacts/multiclass/model.pth",
    "artifacts/binary/model.pth",
]

# Track B: measured power-loss regressor (DeepSolarEye). Optional — falls back to
# the classical illustrative estimate when this checkpoint is absent.
_SEVERITY_MODEL = "artifacts/severity/model.pth"


def load_model(model_path: str | None) -> str:
    """Load a model bundle into STATE. Returns a human-readable status string."""
    if model_path is None:
        model_path = next((p for p in _CANDIDATE_MODELS if Path(p).exists()), None)
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


def analyze(image: Image.Image):
    """Run the full pipeline on one PIL image for the Gradio callback."""
    if image is None:
        return {}, None, None, "Upload a solar-panel image to begin."

    # --- Classical soiling estimate (always available) ---
    soil = estimate_soiling(image)
    soil_overlay = overlay_mask(image, soil["mask"])

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

    summary = (
        f"**Soiling index:** {soil['soiling_index']:.3f}  ·  "
        f"**severity:** {soil['severity']}  ·  "
        f"**coverage ≈** {soil['coverage_pct']:.1f}%\n\n"
        f"{power_line}\n\n"
        f"_Soiling index is an unsupervised relative measure (desaturation vs a "
        f"clean reference); see DATASET.md / README for its limits._"
    )

    model, bundle = STATE["model"], STATE["bundle"]
    if model is None:
        return {}, None, soil_overlay, "**No trained model loaded.**\n\n" + summary

    device = STATE["device"]
    cam, idx, _ = compute_cam(model, bundle, image, device)
    probs = {c: float(p) for c, p in _class_probs(model, bundle, image, device).items()}
    cam_overlay = overlay_cam(image, cam, bundle["img_size"])
    head = f"**Prediction:** {bundle['class_names'][idx]}  ({bundle['task']} model)\n\n"
    return probs, cam_overlay, soil_overlay, head + summary


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
            "Upload a panel photo to get its **condition** (CNN), a **Grad-CAM** "
            "explanation, and a **classical soiling estimate**."
        )
        status = gr.Markdown(load_model(STATE.get("_model_path")))
        load_regressor_into_state()
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Solar panel image")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                label = gr.Label(label="Predicted condition (probabilities)")
                summary = gr.Markdown()
        with gr.Row():
            cam_img = gr.Image(label="Grad-CAM (where the model looked)")
            soil_img = gr.Image(label="Soiling coverage (classical)")
        btn.click(analyze, inputs=inp, outputs=[label, cam_img, soil_img, summary])
        inp.change(analyze, inputs=inp, outputs=[label, cam_img, soil_img, summary])
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Launch the solar-panel soiling demo.")
    ap.add_argument("--model", default=None, help="Path to a model.pth bundle.")
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    STATE["_model_path"] = args.model
    build_demo().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
