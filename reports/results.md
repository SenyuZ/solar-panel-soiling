# Results

Reproduce everything with the commands in each section. The modular binary numbers below are
from an actual run on this machine (Apple-Silicon MPS, fp32) over the curated multi-source
set in `Data/curated/` (Kaggle Dust-Detection + pythonafroz + SolNet 001/002, de-duplicated
and outlier-filtered, see `DATASET.md`). The original Colab figure (A100, AMP) is shown for
context. Extension rows need their datasets, see `DATASET.md`.

## Binary: clean vs dirty (test set, n = 348)

| Model | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|
| ResNet-50 (curated multi-source, watermarks removed, MPS) | 0.851 | 0.862 | 0.783 | 0.821 | 0.696 |
| ResNet-50 (curated *with* watermarks, MPS) | 0.889 | 0.867 | 0.892 | 0.880 | 0.776 |
| ResNet-50 (earlier raw *Dust Detection* only, MPS) | 0.842 | 0.809 | 0.814 | 0.811 | 0.675 |
| ResNet-50 (original Colab, merged 3-dataset, A100) | 0.901 | 0.854 | 0.908 | 0.881 | 0.797 |
| Classical SVM (HSV + GLCM/LBP + edges) | 0.758 | 0.710 | 0.714 | 0.712 | 0.504 |
| Classical Random Forest | 0.738 | 0.708 | 0.634 | 0.669 | 0.455 |

Per-class (cleaned curated ResNet-50): clean P/R/F1 = 0.843 / 0.903 / 0.872 · dirty = 0.862
/ 0.783 / 0.821. Training early-stopped at epoch 12.

Reading the table: this is a data-quality story with an honest twist. Pooling three sources
into a curated set first showed 0.880 F1, but an OCR audit (`scripts/triage_suspects.py`)
found 186 images (about 7%) carrying tiled stock-photo watermarks (Shutterstock/Dreamstime),
a spurious cue. Removing the watermarks (plus 65 near-duplicates) lowered the in-domain F1 to
0.821: those easy, watermarked images had been inflating the benchmark. So the headline
in-domain number is now back near the raw-set level (0.811), but that is the honest, harder
number, and the real test is generalisation, where this model is strong (0.952 OOD F1,
below). The leak-free, directly comparable signal is the deep-vs-classical gap on the cleaned
split: ResNet-50's 0.821 F1 against the best classical 0.712 (SVM), so learned features still
add about 11 F1 points over hand-crafted colour/texture/edge descriptors. On this data the
main classical separable cue is saturation (dust greys panels out); see
`solarsoil/severity.py`.

![Confusion matrix](figures/binary_confusion_test.png)

```bash
python -m solarsoil.train             --config configs/binary.yaml
python -m solarsoil.evaluate          --model artifacts/binary/model.pth --manifest manifests/binary_manifest.csv --split test
python -m solarsoil.models.classical  --config configs/classical.yaml --model-type svm
python -m solarsoil.models.classical  --config configs/classical.yaml --model-type rf
```

## Out-of-distribution (OOD) generalisation

The curated binary model, without any retraining, evaluated on a fully external test set
never seen in training: the Roboflow *"solar panel dirt det"* dataset (2,485 panel-filling
photos, a different source/country/camera; CC BY 4.0). Object-detection boxes are collapsed
to image-level labels (any Low/High-Dirty box → dirty, else clean). A perceptual-hash check
found 0 / 2,485 near-duplicates against the training data, so this is leak-free.

| Test set | n | Accuracy | F1 (dirty) | Macro-F1 | MCC |
|---|---|---|---|---|---|
| In-domain (cleaned curated test split) | 348 | 0.851 | 0.821 | 0.846 | 0.696 |
| External OOD (Roboflow) | 2,485 | 0.930 | 0.952 | 0.912 | 0.835 |

Per-class OOD: clean P/R = 0.787 / 0.980 · dirty P/R = 0.993 / 0.914.

Why this matters: the model scores much higher OOD than in-domain, on images that are
panel-filling with almost no background, exactly the regime where the background-shortcut
(README Limitations) cannot help. This is direct evidence the model learned real dust/panel
features and generalises across sources; the background reliance is a wide-scene artefact,
not a crutch. Notably, removing the in-domain watermarks dropped the in-domain F1 (0.880 →
0.821) but left OOD essentially unchanged (0.955 → 0.952), confirming the watermarks were
inflating the in-domain number, not the model's real ability. This OOD test replaces an
earlier within-Kaggle pythonafroz cross-check (that source is now part of training).

```bash
# Reproduce (needs the Roboflow dataset downloaded to Data/raw/roboflow_dirt — see DATASET.md):
python scripts/roboflow_ood_manifest.py --root Data/raw/roboflow_dirt --out manifests/ood_roboflow_manifest.csv
python -m solarsoil.evaluate --model artifacts/binary/model.pth --manifest manifests/ood_roboflow_manifest.csv --split test
```

## Multi-class & condition (pythonafroz 6-class set, n_test = 133)

| Task | Backbone | Accuracy | Macro-F1 | MCC |
|---|---|---|---|---|
| 6-class (Clean/Dusty/Bird-drop/Snow/Physical/Electrical) | ResNet-50 | 0.842 | 0.857 | 0.809 |
| 3-way condition (clean / soiled / damaged) | ResNet-50 | 0.895 | 0.883 | 0.818 |

Per-class F1 (6-class): Snow-Covered 0.95, Physical-Damage 0.90, Electrical-damage 0.88,
Bird-drop 0.86, Clean 0.79, Dusty 0.77. Despite heavy imbalance (only 48 Physical-Damage
training images), weighted cross-entropy keeps the rare fault classes strong (macro-F1
0.857). The 3-way condition model (the soiling-vs-fault framing) reaches macro-F1 0.883. Best
val-F1: multiclass 0.908 @ epoch 23, condition 0.907 @ epoch 9.

![6-class confusion matrix](figures/multiclass_confusion_test.png)

```bash
python -m solarsoil.data.download --source faulty
python -m solarsoil.data.manifest --data-root Data/raw/faulty/Faulty_solar_panel --out manifests/multiclass_manifest.csv --label-space multiclass
python -m solarsoil.train     --config configs/multiclass.yaml
python -m solarsoil.evaluate   --model artifacts/multiclass/model.pth --manifest manifests/multiclass_manifest.csv --split test
```

## Severity / coverage (Phase 4)

* Track A (classical, runs now): the saturation-based soiling index orders dusty above clean
  on average (sampled panels: clean ≈ 0.10–0.32, dusty ≈ 0.21–0.47). Absolute coverage is
  approximate, since clean and dusty overlap in low-level cues, which is exactly why the CNN
  is worthwhile. Track A is best used as a relative index + localisation aid (red overlay
  below).
* Track B (DeepSolarEye, real run): a 1-output CNN regressor (MSE) trained on the measured %
  power loss of 45,721 real panel images. A quick run (ResNet-18, 12k stratified subset, 8
  epochs, MPS) predicts power loss with MAE ≈ 0.075, on average within about 7.5 percentage
  points of the true value (RMSE 0.110; best val MSE 0.009). More epochs or the full 45k set
  would tighten it further.

```bash
python -m solarsoil.data.download --source deepsolareye      # ~864 MB, auto-extracts
python -m solarsoil.models.regression manifest \
    --data-root Data/raw/deepsolareye/extracted/Solar_Panel_Soiling_Image_dataset/PanelImages \
    --out manifests/deepsolareye_manifest.csv
python -m solarsoil.models.regression train --manifest manifests/deepsolareye_manifest.csv \
    --backbone resnet18 --limit 12000 --epochs 8
```

![Soiling overlay](figures/coverage/Imgdirty_0_1_soiling.png)

```bash
python -m solarsoil.severity --image Data/Dusty --limit 25 --out-dir reports/figures/coverage
```

## Soiling segmentation, measured coverage (U-Net)

A U-Net with an ImageNet-pretrained ResNet-34 encoder, trained on 1,279 image/dust-mask pairs
(val 160, test 160), turns Track A's approximate index into a measured dust coverage % and a
pixel mask. Test Dice 0.47 · IoU 0.38, where the pretrained encoder lifts a from-scratch
U-Net's 0.38 / 0.30. Dust segmentation is genuinely hard (diffuse, fuzzy boundaries plus
noisy pseudo-labels cap it), but the estimate is now well-localised: a clean panel reads
0.0%, the dusty example 12.8% (truth 16.6%; the from-scratch net over-predicted about 28%),
and the mask traces the soiled *left* side rather than a crude top band. More epochs or
cleaner human labels would push it further.

![Predicted dust mask (measured coverage)](figures/segmentation/Imgdirty_0_1_dustseg.png)

```bash
python -m solarsoil.data.download --source dust_seg
python -m solarsoil.models.segmentation manifest --images Data/raw/dust_seg/images --masks Data/raw/dust_seg/masks --out manifests/segmentation_manifest.csv
python -m solarsoil.models.segmentation train --config configs/segmentation.yaml
```
