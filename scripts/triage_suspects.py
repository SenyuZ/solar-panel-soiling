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
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    device = get_device(args.device)
    model, bundle = load_model(args.model, device)
    classes = bundle["class_names"]

    df = pd.read_csv(args.manifest)
    paths = df["filepath"].tolist()
    labels = df["label"].tolist()

    preds, confs = _model_scores(model, bundle, paths, repo_root, device)
    dupes = _phash_dupes(paths, repo_root, args.phash_dist)

    reader = _ocr_reader() if args.ocr else None

    rows = []
    for p, lab, pred, conf in zip(paths, labels, preds, confs):
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
        if reader is not None:
            try:
                if reader.readtext(str(repo_root / p), detail=0):
                    reasons.append("has_text")
            except Exception:
                pass
        if reasons:
            rows.append(
                {"filepath": p, "label": lab, "pred": pred_name,
                 "confidence": round(conf, 3), "reasons": ";".join(reasons),
                 "near_dupe_of": dupes.get(p, [""])[0] if p in dupes else ""}
            )

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
