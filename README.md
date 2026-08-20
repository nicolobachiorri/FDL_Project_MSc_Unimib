# Breast Ultrasound Image Classification: CNN from Scratch vs Transfer Learning

Project for the **Foundations of Deep Learning (2025/2026)** course — Università degli Studi di Milano-Bicocca.

**Authors:** Nicolò Bachiorri, Davide Francesco Caramia, Hervé Mottaran

## Context

Breast cancer screening increasingly relies on ultrasound imaging as a fast, radiation-free complement to mammography, particularly for younger patients and dense breast tissue. Automating the initial triage of ultrasound images into clinically meaningful categories — **benign**, **malignant**, **normal** — could support radiologists in prioritising cases, but it is a genuinely hard problem: ultrasound images are noisy, operator-dependent, and public annotated datasets are small.

The **BUSI** dataset (Al-Dhabyani et al., 2020) reflects this real-world setting: 780 images collected from 600 female patients aged 25–75, acquired in 2018, categorised into the three classes above.

## Objective

This project asks a specific question: **can a CNN trained from scratch reliably classify breast ultrasound images into these three classes, and what are the practical limits of this approach on a small, imbalanced medical imaging dataset?**

The CNN-from-scratch model is the primary contribution of this work. Transfer learning is used only as a comparative reference point, to quantify — not to substitute — the ceiling reached by the from-scratch approach.

The clinical stakes motivate the choice of metrics: misclassifying a malignant mass as benign (false negative) delays treatment, while misclassifying normal tissue as malignant (false positive) leads to unnecessary interventions. Both error types have direct consequences for patients, so per-class F1 and confusion matrices matter as much as overall accuracy.

## Notebook structure

The notebook is organised into four blocks:

1. **Data pipeline** — loading BUSI from Drive, stratified 70/15/15 split, exploratory class-distribution analysis, class weight computation.
2. **CNN from scratch** — baseline architecture iterated across five versions (V1-V5), with progressively refined strategies to handle class imbalance (class weighting, data augmentation, oversampling).
3. **Transfer learning** — four ImageNet-pretrained networks used as feature extractors (frozen base + custom classifier head): VGG16, EfficientNetB0, DenseNet121, MobileNetV3Small.
4. **Final evaluation on the test set** — comparison between the best from-scratch model and the best transfer learning model, with classification report, confusion matrix, and ROC curves.

## Dataset

The BUSI dataset is not included in this repository due to its size. It is publicly available:

- Al-Dhabyani W., Gomaa M., Khaled H., Fahmy A. (2020). *Dataset of breast ultrasound images*. Data in Brief.
- Source: [Kaggle — Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

To reproduce the experiments, download the dataset and recreate the split (stratified 70/15/15) following the pipeline in Block 1 of the notebook.

## Repository structure

```
.
├── FDL_project_bachiorri_caramia_mottaran.ipynb   # full notebook
├── FDL_project_bachiorri_caramia_mottaran.pdf      # PDF report
└── figures/                                        # loss, accuracy, ROC, and confusion matrix plots
```

Folders containing the dataset and trained models (`BUSI_split`, `aug_BUSI_split`, `oversample_BUSI_split`, `Dataset_BUSI_with_GT`, `models`) are excluded from the repository (see `.gitignore`) due to their size.

## Tech stack

Python, TensorFlow/Keras, scikit-learn, developed on Google Colab.
