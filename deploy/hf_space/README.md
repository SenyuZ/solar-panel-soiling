---
title: Solar Panel Soiling Analyzer
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
python_version: "3.12"
pinned: false
license: mit
---

# Solar Panel Soiling Analyzer

Drop a photo of a photovoltaic panel and get:

- **Condition** — clean vs dirty (ResNet-50 classifier) with class probabilities
- **Grad-CAM** — a heatmap of where the model looked
- **Soiling coverage** — a classical image-processing overlay + soiling index
- **Power loss** — the measured DeepSolarEye Track-B regressor (test MAE ≈ 0.075)

Runs on a free CPU Space, so the first request after the Space wakes from sleep
takes a few seconds. Inference is then ~1–2 s per image.

Source code, methodology, datasets, and honest limitations:
**https://github.com/SenyuZ/solar-panel-soiling**

> The model attends to image background in some wide, scene-heavy shots
> (a known shortcut documented in the repo); it is most reliable on
> panel-filling photos. The soiling index is an unsupervised *relative*
> measure, not a calibrated physical quantity.
