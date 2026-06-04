from solarsoil import metrics as M


def test_binary_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    m = M.compute_metrics(y_true, y_pred, ["clean", "dirty"])
    assert m["accuracy"] == 0.75
    assert set(m["per_class"]) == {"clean", "dirty"}
    assert 0.0 <= m["f1"] <= 1.0


def test_multiclass_metrics():
    classes = ["clean", "soiled", "damaged"]
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 1]
    m = M.compute_metrics(y_true, y_pred, classes)
    assert len(m["per_class"]) == 3
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert m["n_samples"] == 6
