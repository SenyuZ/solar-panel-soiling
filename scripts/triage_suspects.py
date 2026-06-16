"""Flag likely-bad images in a manifest for manual review (curation aid).

This is a *non-destructive* triage tool: it never deletes anything. It scores every
image with cheap, reliable signals, copies the flagged ones into a review folder
grouped by reason, and writes a CSV. You then eyeball the folders, delete the truly
bad images from the data root, rebuild the manifest, and retrain.

Signals
-------
* **model_wrong**     — the trained classifier confidently disagrees with the label
                        (p(other class) >= --wrong-thresh). Catches mislabels and
                        images the panel features can't explain (often watermarks,
                        irrelevant subjects, background-dominated shots).
* **model_uncertain** — max class probability < --uncertain-thresh (ambiguous image).
* **near_duplicate**  — perceptual-hash (pHash) Hamming distance <= --phash-dist to
                        another image (residual near-dupes the SHA/phash dedup missed).
* **has_text**        — (optional, needs ``easyocr``) OCR finds text on the image:
                        the most direct watermark/stock-label signal. Enable with --ocr.

Usage
-----
    python scripts/triage_suspects.py \
        --model artifacts/binary/model.pth \
        --manifest manifests/binary_manifest.csv \
        --out reports/review
    # with watermark OCR (pip install easyocr first):
    python scripts/triage_suspects.py ... --ocr
"""
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
import pandas as pd
import torch
from PIL import Image

from solarsoil.data.datasets import build_transforms
from solarsoil.predict import load_model
from solarsoil.utils import get_device


@torch.no_grad()
def _model_scores(model, bundle, paths, repo_root, device, batch_size=64):
    """Return (pred_idx, max_prob) lists aligned 1:1 with ``paths``.

    Unreadable images get (-1, 0.0). Results stay in input order regardless of
    which images load.
    """
    tfm = build_transforms(bundle["img_size"], train=False)
    preds = [-1] * len(paths)
    confs = [0.0] * len(paths)
    batch_tensors, batch_idx = [], []

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        p = model(x).softmax(1).cpu()
        for slot, row in zip(batch_idx, p):
            preds[slot] = int(row.argmax())
            confs[slot] = float(row.max())
        batch_tensors.clear()
        batch_idx.clear()

    for i, p in enumerate(paths):
        try:
            with Image.open(repo_root / p) as im:
                batch_tensors.append(tfm(im.convert("RGB")))
                batch_idx.append(i)
        except Exception:  # unreadable image stays (-1, 0.0) -> flagged
            continue
        if len(batch_tensors) == batch_size:
            flush()
    flush()
    return preds, confs


def _phash_dupes(paths, repo_root, max_dist):
    """Map each path to a list of near-duplicate partner paths (pHash Hamming<=max_dist)."""
    hashes = {}
    for p in paths:
        try:
            with Image.open(repo_root / p) as im:
                hashes[p] = imagehash.phash(im.convert("RGB"))
        except Exception:
            continue
    items = list(hashes.items())
    dupes: dict[str, list[str]] = defaultdict(list)
    for i in range(len(items)):
        pi, hi = items[i]
        for j in range(i + 1, len(items)):
            pj, hj = items[j]
            if (hi - hj) <= max_dist:
                dupes[pi].append(pj)
                dupes[pj].append(pi)
    return dupes


def _ocr_reader():
    import easyocr  # noqa: PLC0415 — optional dependency

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _ocr_has_text(reader, path, max_dim=1000, min_conf=0.4):
    """True if OCR finds confident text. Downscales first for speed (watermarks
    are large enough to survive). Returns the detected strings too."""
    import numpy as np  # noqa: PLC0415

    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = max_dim / max(im.size)
        if scale < 1:
            im = im.resize((int(im.width * scale), int(im.height * scale)))
        arr = np.asarray(im)
    results = reader.readtext(arr, detail=1, canvas_size=max_dim, mag_ratio=1.0)
    texts = [t for _, t, c in results if c >= min_conf and len(t.strip()) >= 2]
    return texts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Triage likely-bad images for manual review.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="reports/review")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--wrong-thresh", type=float, default=0.80)
    ap.add_argument("--uncertain-thresh", type=float, default=0.60)
    ap.add_argument("--phash-dist", type=int, default=6)
    ap.add_argument("--ocr", action="store_true", help="Flag text/watermarks via easyocr.")
    ap.add_argument("--ocr-skip-source", default="",
                    help="Comma-separated source codenames to skip for OCR "
                         "(e.g. clean sources like solnet_001,solnet_002).")
    ap.add_argument("--ocr-cache", default="reports/ocr_cache.tsv",
                    help="Resumable per-image OCR cache (survives interruptions). "
                         "Delete it to force a fresh OCR pass.")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    device = get_device(args.device)
    model, bundle = load_model(args.model, device)
    classes = bundle["class_names"]

    df = pd.read_csv(args.manifest)
    paths = df["filepath"].tolist()
    labels = df["label"].tolist()
    sources = df["source"].tolist() if "source" in df.columns else [""] * len(df)
    skip_ocr = {s.strip() for s in args.ocr_skip_source.split(",") if s.strip()}

    preds, confs = _model_scores(model, bundle, paths, repo_root, device)
    dupes = _phash_dupes(paths, repo_root, args.phash_dist)

    reader = _ocr_reader() if args.ocr else None

    # Resumable OCR cache: "<filepath>\t<text>" per line ("" text = checked, no text).
    # Written/flushed per image so a kill (laptop sleep) only loses the current image.
    ocr_cache: dict[str, str] = {}
    cache_fh = None
    if reader is not None:
        cache_path = Path(args.ocr_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                fp, _, txt = line.partition("\t")
                ocr_cache[fp] = txt
            print(f"Loaded {len(ocr_cache)} cached OCR results from {cache_path}")
        cache_fh = open(cache_path, "a")

    rows = []
    for i, (p, lab, src, pred, conf) in enumerate(zip(paths, labels, sources, preds, confs)):
        pred_name = classes[pred] if 0 <= pred < len(classes) else "UNREADABLE"
        reasons = []
        if pred_name == "UNREADABLE":
            reasons.append("unreadable")
        else:
            if pred_name != lab and conf >= args.wrong_thresh:
                reasons.append("model_wrong")
            if conf < args.uncertain_thresh:
                reasons.append("model_uncertain")
        if p in dupes:
            reasons.append("near_duplicate")
        ocr_text = ""
        if reader is not None and src not in skip_ocr:
            if p in ocr_cache:
                ocr_text = ocr_cache[p]
            else:
                try:
                    texts = _ocr_has_text(reader, repo_root / p)
                    ocr_text = " | ".join(texts)[:120] if texts else ""
                except Exception:
                    ocr_text = ""
                cache_fh.write(f"{p}\t{ocr_text}\n")
                cache_fh.flush()
                if (i + 1) % 100 == 0:
                    print(f"  OCR progress: {i + 1}/{len(paths)}", flush=True)
            if ocr_text:
                reasons.append("has_text")
        if reasons:
            rows.append(
                {"filepath": p, "label": lab, "pred": pred_name,
                 "confidence": round(conf, 3), "reasons": ";".join(reasons),
                 "ocr_text": ocr_text,
                 "near_dupe_of": dupes.get(p, [""])[0] if p in dupes else ""}
            )

    if cache_fh is not None:
        cache_fh.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flagged = pd.DataFrame(rows)
    flagged.to_csv(out / "suspects.csv", index=False)

    # Copy flagged images into per-reason folders for quick visual review.
    for r in rows:
        for reason in r["reasons"].split(";"):
            dest = out / reason
            dest.mkdir(parents=True, exist_ok=True)
            src = repo_root / r["filepath"]
            if src.exists():
                shutil.copy2(src, dest / src.name)

    by_reason = defaultdict(int)
    for r in rows:
        for reason in r["reasons"].split(";"):
            by_reason[reason] += 1
    print(f"Flagged {len(rows)}/{len(paths)} images. Wrote {out/'suspects.csv'}")
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:16s} {n}")
    print(f"Review the copies under {out}/<reason>/, then delete confirmed-bad "
          f"images from the data root and rebuild the manifest.")


if __name__ == "__main__":
    main()
