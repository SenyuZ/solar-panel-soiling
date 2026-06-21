# Datasets, provenance & attribution

> Images are not redistributed in this repository. Each source dataset keeps its own
> licence and terms. Use the download helpers (`python -m solarsoil.data.download
> --source ...`) to fetch them locally into `Data/` (git-ignored). Only metadata
> (manifests) is committed.

Provenance below is taken from the original Team 7 project report (UTS, *Image Processing
and Pattern Recognition*, Assignment 2, Oct 2025).

## Binary training set (`Data/curated/`), curated, multi-source

The binary clean/dirty model is trained on a curated, de-duplicated multi-source set:
1,305 clean / 1,015 dirty = 2,320 images pooled from three public sources, de-duplicated
(SHA-256 + perceptual hash), manually outlier-filtered, and then watermark- and
near-duplicate-cleaned (see below). Per-image provenance is recorded in the `source` column
of `manifests/binary_manifest.csv`:

| # | Dataset | Codename(s) | Link | Images in final set |
|---|---|---|---|---|
| 1 | Kaggle Solar Panel Dust Detection (Garladinne, 2022) | `detect_solar_dust` | [kaggle.com/…/hemanthsai7/solar-panel-dust-detection](https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection) | 1,786 |
| 2 | Kaggle Solar Panel Images (pythonafroz) | `faulty_solar_panel` | [kaggle.com/…/pythonafroz/solar-panel-images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images) | 157 |
| 3 | SolNet (Onim et al., 2023; photos from Bangladesh) | `solnet_001`, `solnet_002` | [mdpi.com/1996-1073/16/1/155](https://www.mdpi.com/1996-1073/16/1/155) · [code](https://github.com/Onimee58/SolNET) | 203 + 174 = 377 |

> Watermark and duplicate cleaning. An OCR/triage pass (`scripts/triage_suspects.py --ocr`)
> flagged 186 images carrying tiled stock-photo watermarks (Shutterstock, Dreamstime,
> Alibaba), a spurious cue present on both classes (113 dirty / 73 clean). These were
> removed, along with 65 near-duplicates (pHash Hamming ≤ 6), taking the set from 2,571 to
> 2,320. This lowered the in-domain test F1 (0.880 → 0.821): the watermarked images had been
> making the benchmark artificially easy. External OOD performance was unaffected (see
> `reports/results.md`).

> Curation is improved but not finished (future work). Inconsistent images were removed both
> by hand (manual outlier filtering) and with tooling, the OCR/triage pass above plus
> perceptual-hash de-duplication. Even so, many inconsistent images remain in the pooled
> multi-source set: mixed framing and scale, wide-angle whole-scene photos where the panel is
> only a small subject, off-subject or low-quality shots, and likely residual near-duplicates
> across the three sources. This lack of standardisation is a known contributor to the
> wide-scene failures documented in the README's Limitations. A further round of cleaning,
> sorting and standardising the set, with tighter framing and dropping off-distribution and
> duplicate images, is future work that should improve both training and the reliability of the
> in-domain benchmark.

> Reproducibility and redistribution. Images are not redistributed. De-duplication is
> scripted (`solarsoil.data.dedup`); the manual outlier and watermark removal is not fully
> scripted, so the exact membership is pinned by the committed `manifests/binary_manifest.csv`
> rather than regenerable from scratch. (The original Team 7 report described a larger
> ~3,539-image curation pass; the set used here is a tighter ~2,320-image re-curation.)

> Earlier raw baseline (for comparison). A previous version trained only on the raw Kaggle
> Dust-Detection set (1,493 clean / 1,069 dirty = 2,562, before curation) and scored 0.811 F1
> / 0.675 MCC. The curated and cleaned multi-source set scores 0.821 F1 / 0.696 MCC in-domain
> but generalises far better out-of-distribution (0.952 F1, see below). Note pythonafroz is
> now folded into training, so it can no longer serve as an independent OOD test; the external
> Roboflow set does instead.

### Note for the multi-class extension
Source #2, Kaggle Solar Panel Images (pythonafroz), is natively a 6-class dataset (Clean,
Dusty, Bird-drop, Electrical-damage, Physical-damage, Snow-Covered) which the original
project collapsed to binary. The multi-class / condition extension recovers those labels by
re-downloading it (`download --source faulty`) and using its native class folders.

## Extension dataset (severity, Track B)

| Purpose | Dataset | Access | Citation |
|---|---|---|---|
| Severity + power loss | DeepSolarEye, 45,754 images labelled with measured % power loss, irradiance, timestamps | `download --source deepsolareye` (Google Drive) | Mehta et al., WACV 2018 |

## Out-of-distribution (OOD) test set, binary generalisation

A fully external dataset used only for evaluation (never in training), to test whether the
binary model generalises beyond its training sources.

| Purpose | Dataset | Access | Licence |
|---|---|---|---|
| External OOD test | "solar panel dirt det" (alex-jcvyb), 2,489 panel-filling photos, different source/country/camera, classes Clean / Low-Dirty / High-Dirty | [Roboflow Universe](https://universe.roboflow.com/alex-jcvyb/solar-panel-dirt-det) | CC BY 4.0 |

Download (needs a free Roboflow account + API key, and `pip install roboflow`):

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
rf.workspace("alex-jcvyb").project("solar-panel-dirt-det").version(1).download(
    "coco", location="Data/raw/roboflow_dirt")
```

Then build an image-level binary manifest and evaluate:

```bash
python scripts/roboflow_ood_manifest.py --root Data/raw/roboflow_dirt --out manifests/ood_roboflow_manifest.csv
python -m solarsoil.evaluate --model artifacts/binary/model.pth --manifest manifests/ood_roboflow_manifest.csv --split test
```

The COCO detection boxes are collapsed to image level: an image is dirty if it has any
Low/High-Dirty box, else clean (`scripts/roboflow_ood_manifest.py`). A perceptual-hash check
confirmed 0 / 2,485 near-duplicates against the training data (leak-free). Result: 0.952 F1 /
MCC 0.835, see `reports/results.md`.

## Segmentation dataset (soiling coverage)

| Purpose | Dataset | Access | Notes |
|---|---|---|---|
| Dust segmentation (exploratory) | Solar Panel Dust Segmentation, 1,604 image + binary dust-mask pairs (0=background, 1=dust) | `download --source dust_seg` | Third-party set (Team 7 / Zhengda Peng, Kaggle): CVAT masks + filtered pseudo-labels. On inspection the labels are inconsistent — many mark background, sky, or watermark text rather than soiling — so the U-Net segmentation is kept exploratory; reliable results would need hand-drawn masks (see the README limitations). |

## Citations

```bibtex
@article{onim2023solnet,
  title   = {SolNet: A Convolutional Neural Network for Detecting Dust on Solar Panels},
  author  = {Onim, Md Saif Hassan and Sakif, Zubayar Mahatab Md and Ahnaf, Adil and
             Kabir, Ahsan and Azad, Abul Kalam and Oo, Amanullah Maung Than and
             Afreen, Rafina and Hridy, Sumaita Tanjim and Hossain, Mahtab and
             Jabid, Taskeed and Ali, Md Sawkat},
  journal = {Energies},
  volume  = {16}, number = {1}, pages = {155}, year = {2023},
  doi     = {10.3390/en16010155}
}

@inproceedings{mehta2018deepsolareye,
  title     = {DeepSolarEye: Power Loss Prediction and Weakly Supervised Soiling
               Localization via Fully Convolutional Networks for Solar Panels},
  author    = {Mehta, Sachin and Azad, Amar P. and Chemmengath, Saneem A. and
               Raykar, Vikas and Kalyanaraman, Shivkumar},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2018}
}
```

Supporting literature cited in the original report: Abuzaid et al. (2022), *Impact of dust
accumulation on photovoltaic panels*, Int. J. Sustainable Eng. 15(1); Bassil et al. (2025),
*Efficient combination of deep learning and tree-based classification models for solar panel
dust detection*, Intelligent Systems with Applications 26.

## Licensing notes
- Code in this repo: MIT (see `LICENSE`).
- Data: governed by each source's own licence/terms (Kaggle dataset licences; SolNet per its
  paper/repo; DeepSolarEye Creative Commons). Review them before any redistribution or
  commercial use. This repo neither relicenses nor redistributes the images.
