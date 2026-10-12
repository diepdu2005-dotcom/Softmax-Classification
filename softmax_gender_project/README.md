# Softmax Gender Classification — Baseline without PCA

## Main learning algorithm
2-class Softmax Regression implemented from scratch with NumPy.

## Dataset
Kaggle: `ashwingupta3012/male-and-female-faces-dataset`

## Pipeline
Dataset audit -> Train/Validation/Test ->
fixed-size grayscale image -> [0,1] pixel scaling -> flatten ->
per-pixel standardization using Train statistics ->
2-class Softmax -> Cross-Entropy -> mini-batch Gradient Descent ->
Validation-based Early Stopping -> final Test evaluation.

## Explicit exclusions from the baseline
- PCA
- HOG
- CNN
- SVM
- other dimensionality-reduction or classification algorithms

## Required plots
- Training Cross-Entropy vs Validation Cross-Entropy
- Training Accuracy vs Validation Accuracy
- Test Confusion Matrix
- Softmax Male-Female pixel weight map
