import torch

from solarsoil.models.cnn import build_model, gradcam_target_layer, split_parameters


def test_build_binary_forward():
    model, _ = build_model("resnet18", num_classes=2, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)


def test_build_multiclass_forward():
    model, _ = build_model("resnet18", num_classes=6, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 6)


def test_split_parameters_partitions_all():
    model, head = build_model("resnet18", num_classes=3, pretrained=False)
    head_p, backbone_p = split_parameters(model, head)
    total = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for p in head_p) + sum(p.numel() for p in backbone_p) == total


def test_gradcam_target_layer_exists():
    model, _ = build_model("resnet18", num_classes=2, pretrained=False)
    assert gradcam_target_layer(model, "resnet18") is not None
