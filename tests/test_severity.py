import numpy as np
from PIL import Image

from solarsoil.severity import estimate_soiling, severity_bucket, soiling_score_map


def test_score_map_in_range():
    rgb = (np.random.default_rng(0).random((128, 128, 3)) * 255).astype("uint8")
    score = soiling_score_map(rgb)
    assert score.min() >= 0.0 and score.max() <= 1.0
    assert score.shape == (128, 128)


def test_estimate_soiling_keys():
    img = Image.fromarray((np.random.default_rng(1).random((96, 96, 3)) * 255).astype("uint8"))
    res = estimate_soiling(img, size=96)
    for key in ("soiling_index", "coverage", "severity", "power_loss_pct", "mask"):
        assert key in res
    assert 0.0 <= res["coverage"] <= 1.0


def test_severity_bucket_monotonic():
    assert severity_bucket(0.0) == "clean"
    assert severity_bucket(0.9) == "heavy"
