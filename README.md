# Solar Panel Soiling & Fault Analysis

Computer-vision toolkit for assessing the condition of photovoltaic (solar)
panels from ordinary photos: is a panel **clean or dirty**, *what kind* of soiling
or fault is present, *how much* of the surface is affected, and *where*.

It started as a University of Technology Sydney (UTS) **Image Processing & Pattern
Recognition** group project (binary clean/dirty classification) and has been
rebuilt into a reproducible package and extended along four axes: multi-class
classification, a classical image-processing baseline, soiling-severity
estimation, and Grad-CAM explainability — plus an interactive demo.

> **Status.** The repository is a clean, installable Python package with a passing
> test suite. The binary CNN pipeline, classical baseline, Grad-CAM, severity
> (Track A) and the demo all run locally (verified on Apple-Silicon MPS). The
> multi-class and DeepSolarEye severity (Track B) paths are implemented and
> config-driven; they require their datasets to be downloaded first (see
> [`DATASET.md`](DATASET.md)).

---

## Highlights

| Capability | What it does | Module |
|---|---|---|
| **Binary classification** | ResNet-50 transfer learning, two-stage fine-tuning (the original task) | `solarsoil.train` / `models.cnn` |
| **Multi-class (hierarchical)** | clean / soiled {dust, bird-drop, snow} / damaged {physical, electrical} | `solarsoil.taxonomy` + configs |
| **Classical IP baseline** | HSV colour + GLCM/LBP texture + edge features → SVM/RF, benchmarked vs the CNN | `features.classical` / `models.classical` |
| **Severity / coverage** | classical soiling index + coverage map (Track A); DeepSolarEye power-loss regression (Track B) | `solarsoil.severity` |
| **Explainability** | from-scratch Grad-CAM heatmaps (also the weak-localisation signal for severity) | `explain.gradcam` |
| **Interactive demo** | drop a photo → prediction + Grad-CAM + soiling overlay | `app/app.py` |

Engineering: config-driven CLIs, fixed seeds, a non-destructive stratified
manifest split, CUDA/MPS/CPU support, best/last checkpoints, metric history
(JSON+CSV), and unit tests.

## Repository layout

```
solar-panel-soiling/
├── src/solarsoil/         # the package
│   ├── data/              # dedup, manifest, datasets, downloads
│   ├── features/          # classical hand-crafted features
│   ├── models/            # CNN backbones + classical SVM/RF
│   ├── explain/           # Grad-CAM
│   ├── severity.py        # soiling coverage / severity (Tracks A & B)
│   ├── taxonomy.py        # binary / condition / multiclass label spaces
│   ├── train.py · evaluate.py · predict.py · engine.py · metrics.py · utils.py
├── configs/               # binary / condition / multiclass / classical / severity
├── app/app.py             # Gradio demo
├── manifests/             # committed split metadata (no images)
├── reports/               # results.md + figures
├── tests/                 # pytest suite
├── DATASET.md             # data provenance, licences, citations
└── pyproject.toml · requirements.txt · LICENSE
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # installs the `solarsoil` package + CLIs
```

## Quickstart (binary, uses the included `Data/`)

```bash
# 1. Build a stratified train/val/test manifest from Data/{Clean,Dusty}
python -m solarsoil.data.manifest --data-root Data --out manifests/binary_manifest.csv --label-space binary

# 2. Train the ResNet-50 baseline (auto-uses CUDA > MPS > CPU)
python -m solarsoil.train --config configs/binary.yaml

# 3. Evaluate on the test split (metrics + confusion matrices)
python -m solarsoil.evaluate --model artifacts/binary/model.pth --manifest manifests/binary_manifest.csv --split test

# 4. Predict on a single image, with a Grad-CAM overlay
python -m solarsoil.predict --model artifacts/binary/model.pth --image Data/Dusty/Imgdirty_0_1.jpg --gradcam

# 5. Classical baseline (image processing + SVM) for comparison
python -m solarsoil.models.classical --config configs/classical.yaml --model-type svm

# 6. Classical soiling/severity estimate (no training needed)
python -m solarsoil.severity --image Data/Dusty --limit 20 --out-dir reports/figures/coverage

# 7. Interactive demo
python app/app.py
```

> Tip: append `--epochs 1 --max-steps 5 --num-workers 0` to `train` for a fast
> smoke run.

## Extensions that need a dataset download

```bash
# Multi-class (6 classes) / condition (clean·soiled·damaged)
python -m solarsoil.data.download --source faulty
python -m solarsoil.data.manifest --data-root Data/raw/faulty --out manifests/multiclass_manifest.csv --label-space multiclass
python -m solarsoil.train --config configs/multiclass.yaml      # or configs/condition.yaml

# Severity Track B (DeepSolarEye, measured % power loss)
python -m solarsoil.data.download --source deepsolareye
```

See [`DATASET.md`](DATASET.md) for sources, licences and citations, and
[`reports/results.md`](reports/results.md) for the results tables.

## Methodology in brief

- **Transfer learning.** ImageNet-pretrained backbone; stage 1 trains only a new
  head (frozen backbone), stage 2 fine-tunes everything at a lower LR. Class
  imbalance handled with weighted cross-entropy.
- **Taxonomy.** Soiling (removable, temporary power loss) and damage (permanent,
  needs repair) are physically different, so the label space is a 2-level
  hierarchy (condition → type) rather than one flat set.
- **Classical baseline.** Interpretable colour/texture/edge descriptors + SVM/RF,
  to quantify how far simple image processing gets versus deep features.
- **Severity.** Track A is an unsupervised, saturation-based soiling index +
  coverage map (transparent but approximate); Track B learns measured power loss
  from DeepSolarEye for a calibrated estimate.
- **Explainability.** Grad-CAM over the last conv block shows *where* the model
  sees soiling/faults.

## Tests

```bash
pytest -q
```

## Acknowledgements & attribution

- Originated as a **UTS Image Processing & Pattern Recognition** group project.
  _TODO: add course code, semester, and teammate credits / contribution note._
- Built on public datasets — see [`DATASET.md`](DATASET.md). Images are **not**
  redistributed here; download scripts fetch them locally.
- DeepSolarEye: Mehta et al., *WACV 2018* (see `DATASET.md` for the citation).

## License

Code: [MIT](LICENSE). Datasets retain their own licences (see `DATASET.md`).
