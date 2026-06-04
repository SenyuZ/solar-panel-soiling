import numpy as np
from PIL import Image

from solarsoil.features.classical import extract_features, feature_names


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8"))


def test_feature_vector_shape_matches_names():
    feats = extract_features(_img())
    names = feature_names()
    assert feats.shape == (len(names),)
    assert feats.dtype == np.float32


def test_features_are_deterministic():
    img = _img(seed=42)
    a = extract_features(img)
    b = extract_features(img)
    assert np.allclose(a, b)
