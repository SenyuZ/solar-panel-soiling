# Datasets, provenance & attribution

> **Images are not redistributed in this repository.** Each source dataset keeps
> its own licence and terms. Use the download helpers
> (`python -m solarsoil.data.download --source ...`) to fetch them locally into
> `Data/` (git-ignored). Only metadata (manifests) is committed.

Provenance below is taken from the original Team 7 project report (UTS, *Image
Processing and Pattern Recognition*, Assignment 2, Oct 2025).

## Binary set shipped with this repo (`Data/`)

The `Data/Clean` (1,493) + `Data/Dusty` (1,069) = **2,562** images in this repo are
the **Kaggle Solar Panel Dust Detection** dataset (raw, before the report's
outlier curation):

- Garladinne, H. S. (2022). *Solar Panel dust detection* [dataset]. Kaggle.
  https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection

## Full multi-source pipeline (the original project)

The project's "multiple datasets" pipeline merged **three** public sources
(referenced by codename in the notebook) and de-duplicated them (SHA-256 +
perceptual hash) into a curated binary set of **3,539** images.

| # | Dataset | Codename(s) | Link | Curated counts |
|---|---|---|---|---|
| 1 | Kaggle **Solar Panel Dust Detection** (Garladinne, 2022) | `detect_solar_dust` | [kaggle.com/…/hemanthsai7/solar-panel-dust-detection](https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection) | 1,358 clean / 878 dirty |
| 2 | Kaggle **Solar Panel Images** (pythonafroz) | `faulty_solar_panel` | [kaggle.com/…/pythonafroz/solar-panel-images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images) | 87 clean / 85 dirty |
| 3 | **SolNet** dataset (Onim et al., 2023; photos from Bangladesh) | `solnet_001`, `solnet_002` | [mdpi.com/1996-1073/16/1/155](https://www.mdpi.com/1996-1073/16/1/155) · [code](https://github.com/Onimee58/SolNET) | 279+389 clean / 380+83 dirty |

(Per-source counts are the de-duplicated curation reported by Team 7; raw totals
were larger.)

### Note for the multi-class extension
Source #2, Kaggle **Solar Panel Images** (pythonafroz), is *natively a 6-class*
dataset — Clean, Dusty, Bird-drop, Electrical-damage, Physical-damage,
Snow-Covered — which the original project collapsed to binary. The multi-class /
condition extension recovers those labels by re-downloading it
(`download --source faulty`) and using its native class folders.

## Extension dataset (severity, Track B)

| Purpose | Dataset | Access | Citation |
|---|---|---|---|
| Severity + power loss | **DeepSolarEye** — 45,754 images labelled with measured % power loss, irradiance, timestamps | `download --source deepsolareye` (Google Drive) | Mehta et al., WACV 2018 |

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

Supporting literature cited in the original report: Abuzaid et al. (2022),
*Impact of dust accumulation on photovoltaic panels*, Int. J. Sustainable Eng.
15(1); Bassil et al. (2025), *Efficient combination of deep learning and
tree-based classification models for solar panel dust detection*, Intelligent
Systems with Applications 26.

## Licensing notes
- **Code** in this repo: MIT (see `LICENSE`).
- **Data**: governed by each source's own licence/terms (Kaggle dataset licences;
  SolNet per its paper/repo; DeepSolarEye Creative Commons). Review them before
  any redistribution or commercial use. This repo neither relicenses nor
  redistributes the images.
