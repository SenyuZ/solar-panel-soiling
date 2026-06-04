# Results

Reproduce everything with the commands in each section. Binary numbers below are
from an actual run on this machine (Apple-Silicon MPS, fp32); the original Colab
figures (A100, AMP) are included for comparison. Extension rows need their
datasets — see `DATASET.md`.

## Binary: clean vs dirty (test set, n = 385)

| Model | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|
| ResNet-50 (original Colab run, A100) | 0.901 | 0.854 | 0.908 | 0.881 | 0.797 |
| **ResNet-50 (modular re-run, MPS)** | **0.842** | 0.809 | 0.814 | **0.811** | 0.675 |
| Classical SVM (HSV + GLCM/LBP + edges) | 0.758 | 0.710 | 0.714 | 0.712 | 0.504 |
| Classical Random Forest | 0.738 | 0.708 | 0.634 | 0.669 | 0.455 |

Per-class (modular ResNet-50): clean P/R/F1 = 0.865 / 0.862 / 0.864 · dirty =
0.809 / 0.814 / 0.811. Training early-stopped at epoch 14; best validation F1
0.865 at epoch 9.

**Reading the table.** The modular re-run lands a few points below the original
Colab run — expected, given a different stratified split + seed, fp32 on MPS (no
AMP), and the original's extra source-specific class weighting. The headline is
the **deep-vs-classical gap**: ResNet-50's 0.811 F1 vs the best classical 0.712
(SVM) — learned features add ~10 F1 points over hand-crafted colour/texture/edge
descriptors. On this data the main *classical* separable cue is saturation (dust
greys panels out); see the discussion in `solarsoil/severity.py`.

![Confusion matrix](figures/binary_confusion_test.png)

```bash
python -m solarsoil.train             --config configs/binary.yaml
python -m solarsoil.evaluate          --model artifacts/binary/model.pth --manifest manifests/binary_manifest.csv --split test
python -m solarsoil.models.classical  --config configs/classical.yaml --model-type svm
python -m solarsoil.models.classical  --config configs/classical.yaml --model-type rf
```

## Multi-class & condition (needs the 6-class source)

| Task | Backbone | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|
| 6-class (multiclass) | ResNet-50 | _to fill_ | | per-class P/R/F1 in eval output |
| 3-way (clean/soiled/damaged) | ResNet-50 | _to fill_ | | the soiling-vs-fault framing |

```bash
python -m solarsoil.data.download --source faulty
python -m solarsoil.data.manifest --data-root Data/raw/faulty --out manifests/multiclass_manifest.csv --label-space multiclass
python -m solarsoil.train     --config configs/multiclass.yaml
python -m solarsoil.evaluate   --model artifacts/multiclass/model.pth --manifest manifests/multiclass_manifest.csv --split test
```

## Severity / coverage (Phase 4)

* **Track A (classical, runs now):** the saturation-based soiling index orders
  dusty above clean on average (sampled panels: clean ≈ 0.10–0.32, dusty ≈
  0.21–0.47). Absolute coverage is approximate — clean and dusty overlap in
  low-level cues, which is exactly why the CNN is worthwhile. Track A is best
  used as a relative index + localisation aid (red overlay below).
* **Track B (DeepSolarEye):** power-loss MAE — _to fill_ once the dataset is
  downloaded and a regression head is trained.

![Soiling overlay](figures/coverage/Imgdirty_0_1_soiling.png)

```bash
python -m solarsoil.severity --image Data/Dusty --limit 25 --out-dir reports/figures/coverage
```
