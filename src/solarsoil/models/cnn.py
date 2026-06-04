"""Transfer-learning CNN backbones with swappable classification heads.

The original project used ResNet-50 (ImageNet V2 weights); this generalises it
so the same training code can benchmark several backbones (Phase 2/3) and so the
Grad-CAM module (Phase 5) can locate the right target layer for each.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

# name -> (constructor, default weights enum)
BACKBONES = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    ),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def _replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Swap the final classification layer for ``num_classes`` and return it."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):  # ResNet family
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.fc
    if hasattr(model, "classifier"):  # MobileNet / EfficientNet
        clf = model.classifier
        if isinstance(clf, nn.Sequential) and isinstance(clf[-1], nn.Linear):
            clf[-1] = nn.Linear(clf[-1].in_features, num_classes)
            return clf[-1]
        if isinstance(clf, nn.Linear):
            model.classifier = nn.Linear(clf.in_features, num_classes)
            return model.classifier
    raise ValueError("Unsupported backbone head; add handling in _replace_head().")


def build_model(backbone: str = "resnet50", num_classes: int = 2, pretrained: bool = True):
    """Construct a backbone with a fresh ``num_classes`` head.

    Returns ``(model, head)`` where ``head`` is the new final module (useful for
    building the feature-extraction-phase optimizer over head params only).
    """
    if backbone not in BACKBONES:
        raise KeyError(f"Unknown backbone {backbone!r}. Options: {list(BACKBONES)}")
    ctor, weights = BACKBONES[backbone]
    model = ctor(weights=weights if pretrained else None)
    head = _replace_head(model, num_classes)
    return model, head


def split_parameters(model: nn.Module, head: nn.Module):
    """Return (head_params, backbone_params) for staged fine-tuning."""
    head_ids = {id(p) for p in head.parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return head_params, backbone_params


def set_trainable(params, trainable: bool) -> None:
    for p in params:
        p.requires_grad = trainable


def gradcam_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    """Return the last spatial conv block, used as the Grad-CAM target layer."""
    if backbone.startswith("resnet"):
        return model.layer4[-1]
    if backbone.startswith("mobilenet") or backbone.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer mapping for backbone {backbone!r}")
