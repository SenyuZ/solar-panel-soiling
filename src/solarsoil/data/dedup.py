"""Near-duplicate image detection (SHA-256 + perceptual hash).

Ported and cleaned up from the original notebook's curation step. The merged
source datasets contained exact and near-duplicate images; training on
duplicates leaks information across train/val/test and inflates scores. This
module finds duplicates so they can be excluded.

Unlike the notebook (which *moved* files as a side effect), these helpers are
non-destructive by default: they report duplicates and let the caller decide.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def sha256(path: str | Path) -> str:
    """Exact-content hash of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def phash(path: str | Path):
    """Perceptual hash of an image (None if it cannot be read)."""
    import imagehash

    try:
        with Image.open(path) as img:
            return imagehash.phash(img)
    except Exception:  # noqa: BLE001 - corrupt/non-image file
        return None


def _fingerprint(path: Path) -> tuple[Path, str | None, object | None]:
    return path, sha256(path), phash(path)


def iter_images(root: str | Path) -> list[Path]:
    """Recursively list image files under ``root``."""
    root = Path(root)
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


@dataclass
class DedupResult:
    unique: list[Path] = field(default_factory=list)
    duplicates: list[Path] = field(default_factory=list)
    unreadable: list[Path] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.unique)} unique, {len(self.duplicates)} duplicates, "
            f"{len(self.unreadable)} unreadable"
        )


def find_duplicates(paths: list[Path], max_workers: int | None = None) -> DedupResult:
    """Partition ``paths`` into unique / duplicate / unreadable.

    A file is a duplicate if it shares an exact SHA-256 *or* a perceptual hash
    with an already-seen image. Processing order is sorted for determinism.
    """
    seen_sha: set[str] = set()
    seen_ph: set[object] = set()
    result = DedupResult()

    fingerprints: dict[Path, tuple[str | None, object | None]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fingerprint, p): p for p in paths}
        for fut in as_completed(futures):
            path, sha, ph = fut.result()
            fingerprints[path] = (sha, ph)

    for path in sorted(paths):  # deterministic iteration
        sha, ph = fingerprints[path]
        if ph is None:
            result.unreadable.append(path)
            continue
        if sha in seen_sha or ph in seen_ph:
            result.duplicates.append(path)
            continue
        seen_sha.add(sha)
        seen_ph.add(ph)
        result.unique.append(path)
    return result


def dedup_directory(root: str | Path) -> DedupResult:
    """Convenience: find duplicates among all images under ``root``."""
    return find_duplicates(iter_images(root))
