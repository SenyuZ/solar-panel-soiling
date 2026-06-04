"""Build a dataset manifest with a reproducible, stratified train/val/test split.

The original notebook physically *moved* files into ``train/``, ``val/`` and
``test/`` folders — destructive and hard to reproduce. Here we instead scan an
image root once and record every image in a single CSV manifest with a ``split``
column. Datasets read from the manifest, so the on-disk image layout is never
mutated and the exact split is captured by the committed CSV + seed.

The same code handles every label space:
* ``binary``     — folders Clean/Dusty  -> clean/dirty
* ``condition``  — 6 source folders      -> clean/soiled/damaged
* ``multiclass`` — 6 source folders      -> the 6 canonical classes

Run as a CLI::

    python -m solarsoil.data.manifest --data-root Data --out manifests/binary_manifest.csv \
        --label-space binary
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .. import taxonomy
from ..utils import setup_logging
from .dedup import IMAGE_EXTS, sha256

logger = logging.getLogger("solarsoil.manifest")

# Known source-dataset prefixes (for best-effort provenance tagging).
_KNOWN_SOURCES = (
    "solnet_001",
    "solnet_002",
    "detect_solar_dust",
    "faulty_solar_panel",
)


def _infer_source(filename: str, default: str) -> str:
    low = filename.lower()
    for src in _KNOWN_SOURCES:
        if low.startswith(src):
            return src
    return default


def scan_images(data_root: str | Path, repo_root: str | Path, label_space: str) -> pd.DataFrame:
    """Scan ``data_root`` (one subfolder per raw class) into a manifest frame.

    Paths are stored relative to ``repo_root`` so the manifest stays portable.
    """
    data_root = Path(data_root)
    repo_root = Path(repo_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    default_source = data_root.name
    rows: list[dict] = []
    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        raw = class_dir.name
        try:
            canonical = taxonomy.normalize_label(raw)
        except KeyError:
            logger.warning("Skipping unrecognised class folder %r", raw)
            continue
        projected = taxonomy.to_label_space(canonical, label_space)
        for img in sorted(class_dir.iterdir()):
            if not (img.is_file() and img.suffix.lower() in IMAGE_EXTS):
                continue
            rel = img.resolve().relative_to(repo_root.resolve()).as_posix()
            rows.append(
                {
                    "id": hashlib.md5(rel.encode()).hexdigest()[:12],
                    "filepath": rel,
                    "label_raw": raw,
                    "label_canonical": canonical,
                    "label": projected,
                    "source": _infer_source(img.name, default_source),
                }
            )
    if not rows:
        raise RuntimeError(f"No images found under {data_root}")
    df = pd.DataFrame(rows)
    logger.info("Scanned %d images across %d classes", len(df), df["label"].nunique())
    return df


def assign_splits(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
    stratify_col: str = "label_canonical",
) -> pd.DataFrame:
    """Add a stratified ``split`` column (train/val/test) in place-safe fashion."""
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be in (0, 1)")
    df = df.reset_index(drop=True)
    strat = df[stratify_col]
    train_idx, tmp_idx = train_test_split(
        df.index, test_size=val_size + test_size, random_state=seed, stratify=strat
    )
    rel_test = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=rel_test, random_state=seed, stratify=strat.loc[tmp_idx]
    )
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def build_manifest(
    data_root: str | Path,
    out_csv: str | Path,
    label_space: str = "binary",
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 123,
    compute_checksum: bool = False,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    """Scan, split and write a manifest CSV. Returns the DataFrame."""
    repo_root = Path(repo_root) if repo_root else Path.cwd()
    df = scan_images(data_root, repo_root, label_space)
    df = assign_splits(df, val_size, test_size, seed)
    if compute_checksum:
        df["checksum"] = [sha256(repo_root / p) for p in df["filepath"]]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    counts = (
        df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(["train", "val", "test"])
    )
    logger.info("Wrote %s\n%s", out_csv, counts.to_string())
    return df


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Build a stratified dataset manifest CSV.")
    ap.add_argument("--data-root", required=True, help="Image root (one subfolder per class).")
    ap.add_argument("--out", required=True, help="Output manifest CSV path.")
    ap.add_argument(
        "--label-space", default="binary", choices=["binary", "condition", "multiclass"]
    )
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--checksum", action="store_true", help="Also store SHA-256 per image (slow).")
    ap.add_argument("--repo-root", default=None, help="Root for relative paths (default: cwd).")
    args = ap.parse_args(argv)

    build_manifest(
        data_root=args.data_root,
        out_csv=args.out,
        label_space=args.label_space,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        compute_checksum=args.checksum,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    main()
