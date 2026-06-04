"""Grad-CAM, implemented from scratch with forward/backward hooks.

Grad-CAM (Selvaraju et al., 2017) weights the last conv layer's activation maps
by the gradient of the target class score, giving a coarse heatmap of *where*
the network looked. For solar panels this visualises which regions drove a
"dirty"/fault prediction — directly answering "what is actually dirty?" — and is
the weak-supervision signal reused for severity localization (Phase 4, Track B).

Implementing it by hand (rather than importing pytorch-grad-cam) keeps the
dependency surface small and makes the mechanism explicit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..data.datasets import build_transforms
from ..models.cnn import gradcam_target_layer


class GradCAM:
    """Compute a Grad-CAM heatmap for a target conv layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fh = target_layer.register_forward_hook(self._save_activations)
        self._bh = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        self._fh.remove()
        self._bh.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """Return (cam HxW in [0,1], class_idx, class_probability)."""
        x = x.clone().requires_grad_(True)  # ensures grads reach the target layer
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        prob = logits.softmax(dim=1)[0, class_idx].item()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, class_idx, prob


def compute_cam(model, bundle: dict, image, device, class_idx: int | None = None):
    """Grad-CAM for one image given a trained model bundle."""
    target = gradcam_target_layer(model, bundle["backbone"])
    cam = GradCAM(model, target)
    try:
        tfm = build_transforms(bundle["img_size"], train=False)
        img = Image.open(image).convert("RGB") if not isinstance(image, Image.Image) else image
        x = tfm(img).unsqueeze(0).to(device)
        model.eval()
        return cam(x, class_idx)
    finally:
        cam.remove()


def overlay_cam(image: Image.Image, cam: np.ndarray, img_size: int, alpha: float = 0.45) -> Image.Image:
    """Blend a CAM heatmap (jet colormap) over the original image."""
    from matplotlib import colormaps

    base = image.convert("RGB").resize((img_size, img_size))
    base_np = np.asarray(base).astype(np.float32) / 255.0
    heat = colormaps["jet"](cam)[..., :3]  # HxWx3 in [0,1]
    blend = (1 - alpha) * base_np + alpha * heat
    return Image.fromarray((blend * 255).clip(0, 255).astype(np.uint8))


def save_gradcam_overlay(model, bundle: dict, image_path, out_path, device, class_idx=None):
    """Compute Grad-CAM and write an overlay PNG. Returns (path, label, prob)."""
    cam, idx, prob = compute_cam(model, bundle, image_path, device, class_idx)
    overlay = overlay_cam(Image.open(image_path), cam, bundle["img_size"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)
    return out_path, bundle["class_names"][idx], prob
