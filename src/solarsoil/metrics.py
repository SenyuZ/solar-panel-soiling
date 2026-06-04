"""Classification metrics and plots, shared by the deep and classical pipelines.

Works for both binary and multi-class label spaces: binary uses the positive
class (``dirty``) for precision/recall/F1, multi-class reports macro averages
plus a full per-class breakdown.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    """Return a dict of headline + per-class metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(class_names)
    average = "binary" if n == 2 else "macro"

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "n_samples": int(len(y_true)),
    }
    rep = classification_report(
        y_true,
        y_pred,
        labels=list(range(n)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    out["per_class"] = {
        c: {
            "precision": float(rep[c]["precision"]),
            "recall": float(rep[c]["recall"]),
            "f1": float(rep[c]["f1-score"]),
            "support": int(rep[c]["support"]),
        }
        for c in class_names
    }
    return out


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str],
    out_path: str | Path,
    normalize: bool = False,
    title: str | None = None,
) -> Path:
    """Render a confusion-matrix heatmap to ``out_path`` (PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion(y_true, y_pred, len(class_names)).astype(float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    size = max(4.0, 0.8 * len(class_names) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else ".0f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or ("Confusion matrix" + (" (normalized)" if normalize else "")))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def format_metrics(metrics: dict, title: str = "Metrics") -> str:
    """Human-readable one-block summary for console/report output."""
    lines = [f"== {title} ==",
             f"accuracy={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
             f"macro_f1={metrics['macro_f1']:.4f}  mcc={metrics['mcc']:.4f}  "
             f"(n={metrics['n_samples']})"]
    if "per_class" in metrics:
        lines.append(f"{'class':<18}{'prec':>7}{'rec':>7}{'f1':>7}{'support':>9}")
        for c, m in metrics["per_class"].items():
            lines.append(
                f"{c:<18}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}{m['support']:>9}"
            )
    return "\n".join(lines)
