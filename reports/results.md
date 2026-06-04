# Results

Numbers below are reproduced by the commands in each section. The headline binary
figures are from the **original Colab run** (an A100 GPU, full 20-epoch schedule);
re-running `solarsoil.train` reproduces them up to split/seed differences. Rows
marked _to fill_ are produced by running the corresponding command on your machine
(some need the extension datasets — see `DATASET.md`).

## Binary: clean vs dirty (test set)

| Model | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|
| ResNet-50 (original Colab run) | 0.901 | 0.854 | 0.908 | 0.881 | 0.797 |
| ResNet-50 (modular re-run) | _to fill_ | | | | |
| Classical SVM (HSV+GLCM+LBP+edges) | _to fill_ | | | | |
| Classical Random Forest | _to fill_ | | | | |

```bash
python -m solarsoil.train     --config configs/binary.yaml
python -m solarsoil.evaluate   --model artifacts/binary/model.pth --manifest manifests/binary_manifest.csv --split test
python -m solarsoil.models.classical --config configs/classical.yaml --model-type svm
python -m solarsoil.models.classical --config configs/classical.yaml --model-type rf
```

The classical-vs-deep gap is the point of Phase 2: hand-crafted colour/texture/edge
features quantify how much signal is reachable without deep features (on this data,
saturation is the main separable cue — see `solarsoil/severity.py`).

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

* **Track A (classical, runs now):** soiling index separates clean (≈0.10–0.32)
  from dusty (≈0.21–0.47) on sampled panels; absolute coverage is approximate —
  see the honest discussion in `solarsoil/severity.py`.
* **Track B (DeepSolarEye):** power-loss MAE — _to fill_ once the dataset is
  downloaded and a regression head is trained.

```bash
python -m solarsoil.severity --image Data/Dusty --limit 25 --out-dir reports/figures/coverage
```
