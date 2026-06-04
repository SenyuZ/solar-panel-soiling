"""solarsoil — solar panel soiling & fault analysis toolkit.

A small, reproducible computer-vision package that grew out of a UTS
Image Processing & Pattern Recognition project. It covers:

* binary clean/dirty classification (ResNet-50 transfer learning),
* multi-class soiling/fault classification (clean / soiled / damaged taxonomy),
* a classical image-processing baseline (colour/texture/edge features + SVM/RF),
* soiling severity / coverage estimation,
* Grad-CAM explainability.

The package is intentionally import-light: heavy dependencies (torch, sklearn)
are imported inside the submodules that need them, so `import solarsoil` and the
classical pipeline work even without a deep-learning stack installed.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
