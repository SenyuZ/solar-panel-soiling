import re

import numpy as np
from PIL import Image

from solarsoil.models.regression import (
    DEFAULT_LABEL_REGEX,
    RegressionDataset,
    build_regression_manifest,
    parse_target,
)


def test_parse_target():
    pat = re.compile(DEFAULT_LABEL_REGEX)
    assert parse_target("solar_Tue_L_0.123_I_0.5.jpg", pat) == 0.123
    assert parse_target("no_label_here.jpg", pat) is None


def _make_dataset(root, n=18):
    rng = np.random.default_rng(0)
    d = root / "ds"
    d.mkdir(parents=True)
    for i in range(n):
        loss = round(float(rng.random()), 3)
        arr = (rng.random((24, 24, 3)) * 255).astype("uint8")
        Image.fromarray(arr).save(d / f"img_{i}_L_{loss}_I_0.5.jpg")
    return d


def test_build_regression_manifest_and_dataset(tmp_path):
    d = _make_dataset(tmp_path)
    out = tmp_path / "reg.csv"
    df = build_regression_manifest(d, out, val_size=0.25, test_size=0.25, repo_root=tmp_path)
    assert "power_loss" in df.columns
    assert set(df["split"]) == {"train", "val", "test"}
    assert out.exists()

    ds = RegressionDataset(df, "train", img_size=24, repo_root=tmp_path)
    x, y = ds[0]
    assert x.shape == (3, 24, 24)
    assert y.shape == (1,)
