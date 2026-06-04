"""Helpers to fetch the source datasets.

We never redistribute images in this repo (see DATASET.md for licensing). These
helpers pull the original public datasets into a local ``Data/raw`` folder so the
pipeline can be reproduced.

Kaggle sources need credentials: set ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` or place
``~/.kaggle/kaggle.json`` (https://www.kaggle.com/docs/api).

DeepSolarEye is distributed via Google Drive from the authors' project page; we
print instructions rather than hard-coding a fragile file id.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from ..utils import setup_logging

logger = logging.getLogger("solarsoil.download")

# Best-effort mapping of this project's source codenames to public datasets.
# Confirm against the project report before citing in DATASET.md.
SOURCES: dict[str, dict] = {
    "faulty": {
        "kaggle": "pythonafroz/solar-panel-images",
        "desc": "6-class clean/dusty/bird-drop/snow/physical/electrical (multi-class source).",
        "maps_to": "faulty_solar_panel",
    },
    "dust": {
        "kaggle": "hemanthsai7/solar-panel-dust-detection",
        "desc": "Binary clean/dusty solar panel images.",
        "maps_to": "detect_solar_dust",
    },
    "deepsolareye": {
        "url": "https://deep-solar-eye.github.io/",
        "desc": "45,754 images labelled with % power loss + irradiance (severity / Track B).",
        "citation": "Mehta et al., DeepSolarEye, WACV 2018.",
    },
}


def download_kaggle(dataset_id: str, dest: str | Path | None = None) -> Path:
    """Download a Kaggle dataset via kagglehub; optionally copy into ``dest``."""
    import kagglehub

    logger.info("Downloading Kaggle dataset %s ...", dataset_id)
    cached = Path(kagglehub.dataset_download(dataset_id))
    logger.info("Downloaded to cache: %s", cached)
    if dest is None:
        return cached
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cached, dest, dirs_exist_ok=True)
    logger.info("Copied to: %s", dest)
    return dest


def fetch(source: str, dest_root: str | Path = "Data/raw") -> Path | None:
    """Fetch one named source into ``dest_root/<source>``."""
    if source not in SOURCES:
        raise KeyError(f"Unknown source {source!r}. Options: {list(SOURCES)}")
    spec = SOURCES[source]
    dest = Path(dest_root) / source
    if "kaggle" in spec:
        return download_kaggle(spec["kaggle"], dest)
    # DeepSolarEye (manual Google Drive download)
    logger.warning(
        "%s is distributed via Google Drive. Visit %s, download the dataset, and\n"
        "extract it to %s . Cite: %s",
        source,
        spec.get("url"),
        dest,
        spec.get("citation", ""),
    )
    return None


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Download source datasets.")
    ap.add_argument("--source", required=True, choices=[*SOURCES.keys(), "all"])
    ap.add_argument("--dest-root", default="Data/raw")
    args = ap.parse_args(argv)

    sources = list(SOURCES) if args.source == "all" else [args.source]
    for src in sources:
        try:
            fetch(src, args.dest_root)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch %s: %s", src, exc)


if __name__ == "__main__":
    main()
