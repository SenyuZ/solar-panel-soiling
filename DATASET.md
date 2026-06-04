# Datasets, provenance & attribution

> **Images are not redistributed in this repository.** Each source dataset keeps
> its own licence and terms. Use the download helpers
> (`python -m solarsoil.data.download --source ...`) to fetch them locally into
> `Data/`, which is git-ignored. Only metadata (manifests) is committed.

## ⚠️ Provenance to be finalised against the project report

The original group project merged **four hand-picked source datasets** (referred
to in the notebook by the codenames below) into a single, de-duplicated binary
`clean` / `dirty` set. The exact origins/licences are documented in the project
report and **should be reconciled with it before publishing**. The mapping below
is this repo's best-effort identification and is marked accordingly.

| Codename (notebook) | Best-effort identification | Confidence | Notes |
|---|---|---|---|
| `faulty_solar_panel` | Kaggle **"Solar Panel Images – Clean and Faulty"** (pythonafroz) — 6 classes: Clean, Dusty, Bird-drop, Electrical-damage, Physical-damage, Snow-Covered | High | Basis for the multi-class / condition extension. |
| `detect_solar_dust` | Kaggle **"Solar Panel dust detection"** (hemanthsai7) — binary clean/dusty | Medium | Largest single contributor to the binary set. |
| `solnet_001` | *Unidentified* | — | Confirm from report. |
| `solnet_002` | *Unidentified* | — | Confirm from report. |

### Consolidated binary set (this repo's `Data/`)
After de-duplication (SHA-256 + perceptual hash, see `solarsoil.data.dedup`):

| Class | Images |
|---|---|
| `Clean` | 1,493 |
| `Dusty` (→ `dirty`) | 1,069 |
| **Total** | **2,562** |

The finer source labels (bird droppings, physical/electrical damage, snow) were
**merged into binary** for the original task. The multi-class extension recovers
them from the `faulty_solar_panel` source (re-download via the `faulty` source).

## Extension datasets

| Purpose | Dataset | Access | Citation |
|---|---|---|---|
| Multi-class / condition | Kaggle "Solar Panel Images – Clean and Faulty" (pythonafroz) | `download --source faulty` | Kaggle dataset page |
| Severity + power loss (Track B) | **DeepSolarEye** — 45,754 images labelled with measured % power loss, irradiance, timestamps | `download --source deepsolareye` (Google Drive, manual) | Mehta et al., *DeepSolarEye: Power Loss Prediction and Weakly Supervised Soiling Localization via Fully Convolutional Networks for Solar Panels*, WACV 2018 |

### Citation (DeepSolarEye)
```bibtex
@inproceedings{mehta2018deepsolareye,
  title={DeepSolarEye: Power Loss Prediction and Weakly Supervised Soiling
         Localization via Fully Convolutional Networks for Solar Panels},
  author={Mehta, Sachin and Azad, Amar P. and Chemmengath, Saneem A. and
          Raykar, Vikas and Kalyanaraman, Shivkumar},
  booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018}
}
```

## Licensing notes
- **Code** in this repo: MIT (see `LICENSE`).
- **Data**: governed by each source's own licence/terms (Kaggle dataset licences,
  DeepSolarEye Creative Commons terms). Review them before any redistribution or
  commercial use. This repo neither relicenses nor redistributes the images.
