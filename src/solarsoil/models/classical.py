"""Classical ML baseline: hand-crafted features + SVM / Random Forest.

Trains on the same manifest split as the CNN so the two are directly comparable
(Phase 2 benchmark). Outputs metrics JSON, a confusion matrix, and a saved model.

Example::

    python -m solarsoil.models.classical --config configs/classical.yaml
    python -m solarsoil.models.classical --config configs/classical.yaml --model-type rf
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .. import metrics as M
from .. import taxonomy
from ..features.classical import extract_dataset
from ..utils import load_config, save_json, set_seed, setup_logging

logger = logging.getLogger("solarsoil.classical")


def build_estimator(model_type: str, seed: int):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_type == "svm":
        from sklearn.svm import SVC

        clf = SVC(C=10.0, gamma="scale", kernel="rbf",
                  class_weight="balanced", random_state=seed)
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type {model_type!r} (svm|rf)")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train_classical(
    manifest: str | Path,
    label_space: str = "binary",
    model_type: str = "svm",
    out_dir: str | Path = "artifacts/classical",
    repo_root: str | Path | None = None,
    size: int = 256,
    limit: int | None = None,
    seed: int = 123,
) -> dict:
    set_seed(seed)
    class_names = taxonomy.label_space(label_space)
    idx = {c: i for i, c in enumerate(class_names)}

    logger.info("Extracting classical features ...")
    X_tr, y_tr = extract_dataset(manifest, "train", repo_root, size, limit)
    X_te, y_te = extract_dataset(manifest, "test", repo_root, size, limit)
    logger.info("Feature matrix: train %s, test %s", X_tr.shape, X_te.shape)

    est = build_estimator(model_type, seed)
    est.fit(X_tr, y_tr)
    pred = est.predict(X_te)

    y_true = [idx[l] for l in y_te]
    y_pred = [idx[l] for l in pred]
    results = M.compute_metrics(y_true, y_pred, class_names)
    print(M.format_metrics(results, title=f"classical-{model_type} / test ({label_space})"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({"model_type": model_type, "label_space": label_space, **results},
              out_dir / f"metrics_{model_type}.json")
    M.plot_confusion_matrix(y_true, y_pred, class_names, out_dir / f"confusion_{model_type}.png")
    try:
        import joblib

        joblib.dump({"estimator": est, "class_names": class_names, "feature_size": size},
                    out_dir / f"classical_{model_type}.joblib")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not pickle estimator: %s", exc)
    logger.info("Wrote classical results to %s", out_dir)
    return results


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Train the classical (SVM/RF) baseline.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--task", default=None, choices=["binary", "condition", "multiclass"])
    ap.add_argument("--model-type", default=None, choices=["svm", "rf"])
    ap.add_argument("--limit", type=int, default=None, help="Cap images/split (smoke test).")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else {}
    manifest = args.manifest or cfg.get("manifest")
    if not manifest:
        ap.error("Provide --manifest or --config with a manifest entry.")
    train_classical(
        manifest=manifest,
        label_space=args.task or cfg.get("task", "binary"),
        model_type=args.model_type or cfg.get("model_type", "svm"),
        out_dir=args.out_dir or cfg.get("out_dir", "artifacts/classical"),
        repo_root=cfg.get("repo_root", Path.cwd()),
        size=int(cfg.get("feature_size", 256)),
        limit=args.limit,
        seed=int(cfg.get("seed", 123)),
    )


if __name__ == "__main__":
    main()
