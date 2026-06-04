import numpy as np
from PIL import Image

from solarsoil.data.datasets import make_dataloaders
from solarsoil.data.manifest import build_manifest


def _make_fake_dataset(root, n_per_class=12):
    rng = np.random.default_rng(0)
    for cls in ["Clean", "Dusty"]:
        d = root / "Data" / cls
        d.mkdir(parents=True)
        for i in range(n_per_class):
            arr = (rng.random((32, 32, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{cls}_{i}.jpg")
    return root / "Data"


def test_build_manifest_splits(tmp_path):
    data_root = _make_fake_dataset(tmp_path)
    out = tmp_path / "manifest.csv"
    df = build_manifest(data_root, out, label_space="binary",
                        val_size=0.25, test_size=0.25, seed=0, repo_root=tmp_path)
    assert out.exists()
    assert set(df["split"]) == {"train", "val", "test"}
    assert set(df["label"]) == {"clean", "dirty"}
    assert len(df) == 24


def test_make_dataloaders(tmp_path):
    data_root = _make_fake_dataset(tmp_path)
    out = tmp_path / "manifest.csv"
    build_manifest(data_root, out, label_space="binary",
                   val_size=0.25, test_size=0.25, seed=0, repo_root=tmp_path)
    data = make_dataloaders(out, label_space="binary", img_size=32,
                            batch_size=4, num_workers=0, repo_root=tmp_path)
    xb, yb = next(iter(data["loaders"]["train"]))
    assert xb.shape[1:] == (3, 32, 32)
    assert data["class_names"] == ["clean", "dirty"]
    assert data["class_weights"].numel() == 2
