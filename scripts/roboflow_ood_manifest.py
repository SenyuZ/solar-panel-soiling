"""Convert the Roboflow 'solar panel dirt det' COCO export into an image-level
binary (clean/dirty) manifest for use as an out-of-distribution (OOD) test set.

Each image carries one detection box (Clean / Low-Dirty / High-Dirty). We collapse
to image-level: an image is *dirty* if it has any Low/High-Dirty box, else *clean*.
Images with no annotation are skipped. All rows get split=test so they can be fed
straight to ``solarsoil.evaluate`` as a held-out OOD set the model never trained on.

    python scripts/roboflow_ood_manifest.py \
        --root Data/raw/roboflow_dirt --out manifests/ood_roboflow_manifest.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

DIRTY = {"Low-Dirty", "High-Dirty"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Data/raw/roboflow_dirt")
    ap.add_argument("--out", default="manifests/ood_roboflow_manifest.csv")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)

    root = Path(args.root)
    repo_root = Path(args.repo_root).resolve()
    rows = []
    for split in ("train", "valid", "test"):
        jf = root / split / "_annotations.coco.json"
        if not jf.exists():
            continue
        d = json.loads(jf.read_text())
        cats = {c["id"]: c["name"] for c in d["categories"]}
        by_img: dict[int, set[str]] = {}
        for a in d["annotations"]:
            by_img.setdefault(a["image_id"], set()).add(cats[a["category_id"]])
        for im in d["images"]:
            labs = by_img.get(im["id"], set())
            if not labs:
                continue  # no annotation -> ambiguous, skip
            label = "dirty" if labs & DIRTY else "clean"
            rel = (root / split / im["file_name"]).resolve().relative_to(repo_root).as_posix()
            rows.append({
                "id": hashlib.md5(rel.encode()).hexdigest()[:12],
                "filepath": rel,
                "label_raw": ";".join(sorted(labs)),
                "label": label,
                "source": "roboflow_dirt_det",
                "split": "test",
            })

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(df)} images "
          f"({(df.label=='clean').sum()} clean / {(df.label=='dirty').sum()} dirty)")


if __name__ == "__main__":
    main()
