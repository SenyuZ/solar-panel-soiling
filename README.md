# Solar Panel Soiling & Fault Analysis

Computer-vision toolkit for assessing the condition of photovoltaic (solar)
panels from ordinary photos: is a panel **clean or dirty**, *what kind* of soiling
or fault is present, *how much* of the surface is affected, and *where*.

It started as a University of Technology Sydney (UTS) **Image Processing & Pattern
Recognition** group project (binary clean/dirty classification) and has been
rebuilt into a reproducible package and extended along several axes: multi-class
classification, a classical image-processing baseline, soiling-severity estimation
and segmentation, and Grad-CAM explainability — plus an interactive demo.

> **Status.** A clean, installable Python package with 22 passing tests. The binary
> and multi-class (6-class + 3-way condition) classifiers, the classical baseline,
> Grad-CAM, severity Track A and the demo all run with **real results** on
> Apple-Silicon MPS (see [`reports/results.md`](reports/results.md)). The DeepSolarEye
> severity regressor (Track B) is trained on the real 45k-image dataset — it predicts
> a panel's power loss to **MAE ≈ 0.075** (~7.5 percentage points). See
> [`DATASET.md`](DATASET.md) for data sources.

---

## Highlights

| Capability | What it does | Module |
|---|---|---|
| **Binary classification** | ResNet-50 transfer learning, two-stage fine-tuning (the original task) | `solarsoil.train` / `models.cnn` |
| **Multi-class (hierarchical)** | clean / soiled {dust, bird-drop, snow} / damaged {physical, electrical} | `solarsoil.taxonomy` + configs |
| **Classical IP baseline** | HSV colour + GLCM/LBP texture + edge features → SVM/RF, benchmarked vs the CNN | `features.classical` / `models.classical` |
| **Severity / coverage** | classical soiling index + coverage map (Track A); DeepSolarEye power-loss CNN regression (Track B) | `severity` · `models.regression` |
| **Soiling segmentation** | **ResNet-34 U-Net** (pretrained encoder) → measured dust coverage % + pixel mask | `models.segmentation` |
| **Explainability** | from-scratch Grad-CAM heatmaps (also the weak-localisation signal for severity) | `explain.gradcam` |
| **Interactive demo** | drop a photo → prediction + Grad-CAM + soiling overlay + measured power-loss (Track B when available) | `app/app.py` |

Engineering: config-driven CLIs, fixed seeds, a non-destructive stratified
manifest split, CUDA/MPS/CPU support, best/last checkpoints, metric history
(JSON+CSV), and unit tests.

## Examples

Binary test set (modular ResNet-50): **84.2% accuracy, 0.811 F1, 0.675 MCC** —
about 10 F1 points above the best classical baseline (SVM, 0.712 F1).
Multi-class (6 classes): **macro-F1 0.857**, with the rare fault classes held up by
class weighting; 3-way clean/soiled/damaged: **macro-F1 0.883**. Full tables and
the deep-vs-classical discussion are in [`reports/results.md`](reports/results.md).

| Grad-CAM — *where the model looks* (multi-class, physical damage) | Soiling coverage — *how much surface is dirty* (classical, Track A) |
|---|---|
| ![Grad-CAM](reports/figures/multiclass_physical_gradcam.png) | ![Soiling](reports/figures/coverage/Imgdirty_0_1_soiling.png) |

> ⚠️ These are **two different things**: Grad-CAM is a coarse *attention* heatmap
> ("where did the classifier look?"), while the soiling overlay is the *coverage*
> estimate ("how much of the surface is dirty?"). A Grad-CAM blob is **not** a dirt
> map — and on the binary model it sometimes lands on the background (see
> **Limitations** below).

A **ResNet-34 U-Net** (ImageNet-pretrained encoder) measures dust coverage directly —
a clean panel reads 0%, this dusty one **12.8%** (truth 17%; test Dice 0.47):

![Measured dust segmentation](reports/figures/segmentation/Imgdirty_0_1_dustseg.png)

## Repository layout

```
solar-panel-soiling/
├── src/solarsoil/         # the package
│   ├── data/              # dedup, manifest, datasets, downloads
│   ├── features/          # classical hand-crafted features
│   ├── models/            # CNN backbones, classical SVM/RF, U-Net segmenter
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

# 4b. Add a measured % power-loss estimate (Track B / DeepSolarEye regressor)
python -m solarsoil.predict --model artifacts/binary/model.pth --image Data/Dusty/Imgdirty_0_1.jpg --power-loss

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
python -m solarsoil.data.manifest --data-root Data/raw/faulty/Faulty_solar_panel --out manifests/multiclass_manifest.csv --label-space multiclass
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

## Limitations & honest caveats

- **The binary model partly exploits background context (shortcut learning).**
  Grad-CAM revealed that on the uncurated, whole-scene Dust-Detection photos, the
  clean/dirty classifier sometimes attends to the *surroundings* (buildings, sky,
  people) rather than the panel. I tested whether this hurts generalisation with a
  **cross-dataset evaluation**: on the *different* pythonafroz tight-crop Clean/Dusty
  set the model still scored **0.95 F1** — so it has clearly learned real panel
  features too, and the shortcut is present but **not** catastrophic. The multi-class
  model (tight panel crops) doesn't show it at all.

  | ✅ Works on tight crops | ❌ Fails on wide scenes |
  |---|---|
  | ![Grad-CAM success — attention on the bird droppings](reports/figures/gradcam_multiclass/Bird%20%2810%29_gradcam.png) | ![Grad-CAM failure — attention on background buildings, not the panel](reports/figures/gradcam/Imgclean_0_0_gradcam.png) |
  | Both hot spots land on the actual bird droppings (multi-class, tight panel crop). | A "clean" prediction driven by the background cityscape/crane, not the panels at the bottom. |

  **Why the wide-scene cases fail (one root cause).** Almost every bad figure in this
  project — the heatmap on a person standing on the array, the coverage overlay
  smeared across sky and buildings, the empty segmentation mask — comes from the
  **uncurated wide-angle Dust-Detection photos**, where the panel occupies a *small
  fraction* of the frame. Both the attention map and the unsupervised desaturation
  then latch onto whatever dominates the image, which is background. The tight-crop
  multi-class images (above) don't show this at all. A related, milder point: on a
  genuinely **clean** panel there is no dirt to localize, so a "find the dirt"
  heatmap or coverage map is ill-posed and will point somewhere arbitrary.

  **Mitigation (future work) — and why standardised data matters.** The fix is to
  **detect and crop to the panel region before classifying**, so only the panel and
  its dirt are in frame. That removes the background the model can cheat on, makes
  the attention maps and coverage overlays meaningful, and would need a panel
  detector or segmentation head (real work, hence future). More broadly, the failures
  here are a **data-quality** story as much as a modelling one: the tight-crop
  pythonafroz and multi-class sets — where every image is *standardised* to a single
  centred panel — give clean attention, honest coverage, and 0.95 cross-dataset F1,
  while the uncurated wide-angle photos do not. Consistent framing, scale, and
  subject isolation at capture time are worth more than extra model capacity; a
  detector/crop step is really a way to *impose* that standardisation after the fact.

- **Grad-CAM is a diagnostic, not a dirt localizer.** It comes from a ~7×7 conv
  grid, so it's always a blurry blob: it answers *where the model looked*, not
  *which pixels are dirty*. That makes it well-suited to **discrete, localized
  faults** (a crack, a clump of bird droppings) but a poor fit for **diffuse soiling**
  (a dust film or snow over the whole panel), where there's no single spot to point
  at and the blob just lands on the highest-contrast patch. Its real value in this
  project is **model introspection** — it is how the background shortcut above was
  discovered. Spatial extent ("how much / where is the dirt") is answered by the
  **segmentation model** and the classical **soiling overlay** (Track A), which are
  the right tools for the job. Sharper CAM variants (Grad-CAM++, Score-CAM, LayerCAM)
  would give less blobby, multi-region maps but remain bounded by the conv
  resolution.

- **Track A's soiling index is unsupervised and approximate** (relative desaturation,
  not calibrated coverage); for trustworthy numbers use the Track B power-loss model.

- **Track B is a quick run** (ResNet-18, 12k-image subset, 8 epochs); training on the
  full 45k set for longer would lower the MAE further.

## Tests

```bash
pytest -q
```

## Acknowledgements & attribution

This work originated as **Team 7's** project for UTS *Image Processing and Pattern
Recognition* (Assignment 2, 2025). Team members: Jiaao Su, Timothy Chan, Md Fardin
Hossain, Zhengda Peng, Senyu Zhu, Anik Chandra Sarkar.

This published repository builds on and extends **Senyu Zhu's** contributions to
that project — the dataset retrieval / de-duplication / curation pipeline and the
ResNet-50 transfer-learning classifier. Other components of the original group
submission (additional CNN/VGG16/InceptionV3 benchmarking and U-Net segmentation
experiments) were led by other team members and are **not** included here. The
multi-class, classical-baseline, severity, Grad-CAM and demo work is new.

Notably, the original report's stated future work — *"localize soiling, quantify
the dirty-area ratio, and trigger cleaning decisions based on thresholds rather
than coarse binary labels"* — directly motivates the severity/coverage extension.

- Data sources, licences and citations: [`DATASET.md`](DATASET.md). Images are
  **not** redistributed here; download scripts fetch them locally.
- DeepSolarEye: Mehta et al., *WACV 2018*; SolNet: Onim et al., *Energies* 2023
  (see `DATASET.md`).

## License

Code: [MIT](LICENSE). Datasets retain their own licences (see `DATASET.md`).
