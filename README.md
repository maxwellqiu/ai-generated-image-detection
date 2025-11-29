# Real vs AI-Generated Image Forensics

This repository contains the code, experiments, and report for a project on detecting **real vs AI-generated images** (face images) using deep convolutional neural networks.

The current baseline is a **ResNet-50 classifier** trained on the **DeepDetect-2025** dataset (Kaggle), with additional tools for visualization (probability distributions, qualitative examples, Grad-CAM).

---

## 🎯 Project Goals

- Build **strong baselines** for real vs AI-generated image classification.
- Provide a clean, reusable **code structure** for future models (e.g., Global+Local models, frequency branch, etc.).
- Generate **publication-ready figures and metrics** (ROC-AUC, PR-AUC, F1, Grad-CAM) for a CV / forensics-oriented paper.

---

## 📁 Repository Structure

```text
ai-generated-image-detection/
├── notebooks/
│   └── baseline_resnet50_real_vs_ai.ipynb   # Baseline ResNet-50 notebook (Kaggle DeepDetect-2025)
│
├── src/
│   ├── dataset.py                           # RealFakeDataset + default transforms
│   ├── models.py                            # Model factory (ResNet-50 baseline, etc.)
│   └── real_vs_ai/                          # (Reserved) training / evaluation scripts
│       ├── __init__.py
│       ├── models/                          # (Reserved) model variants
│       └── train.py                         # (Reserved) script-style training entrypoint
│
├── results/
│   └── figures/
│       └── Real_vs_AI_ResNet50_Baseline_Report.pdf  # Baseline mini-report (PDF)
│       # + confusion matrix, P(fake) histograms, qualitative examples, Grad-CAM, etc.
│
├── paper/
│   └── Real_vs_AI_ResNet50_Baseline_Report.pdf      # Same report (for convenience)
│
├── requirements.txt                        # Minimal Python dependencies
├── .gitignore
└── README.md
