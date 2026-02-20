# Data-Driven Prediction and Classification of Multi-Component Alloys Using Interpretable Machine Learning
## Overview

This repository contains the implementation of a data-driven framework for predicting thermomechanical properties and classifying performance of multi-component metallic alloys using interpretable machine learning.

The framework integrates:

- Multi-output Random Forest regression for tensile strength and melting point prediction
- Logistic regression for performance classification
- PCA and K-Means clustering for compositional family discovery
- Feature importance and coefficient analysis for interpretability

The study is based on a dataset of 2,673 alloy systems described by ~30 elemental composition features.
## 📄 Preprint

The full research manuscript is publicly available at:

DOI: https://doi.org/10.5281/zenodo.XXXXXXX
## Dataset

The dataset used in this study was obtained from publicly available alloy databases and literature sources (via Kaggle).

Each entry contains:

- ~30 elemental composition features (wt.%)
- Ultimate Tensile Strength (UTS)
## Methodology

### 1. Regression (Multi-Output)
Random Forest Regressor was used to predict:
- Ultimate Tensile Strength (UTS)
- Melting Point

Performance metric:
- R² ≈ 0.85 (test set)

### 2. Classification
Logistic Regression classifier categorizes alloys as:
- High Performance
- Standard Performance

Evaluation metrics:
- Accuracy > 80%
- ROC-AUC ≈ 0.87

### 3. Unsupervised Learning
- Principal Component Analysis (PCA)
- K-Means clustering

Used to identify natural compositional alloy families.
## Interpretability

Model transparency was ensured using:

- Random Forest feature importance ranking
- Logistic regression coefficient analysis

Key alloying elements such as Fe, Cr, and Ni were identified as dominant contributors, consistent with established metallurgical principles.
## Deployment

The trained models were serialized using joblib and integrated into a Streamlit-based interface for real-time alloy property prediction and cluster identification.

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py

---

## 📂 Repository Structure

```markdown
## Repository Structure

.
├── app.py
├── rf_regression_model.pkl
├── logistic_classifier.pkl
├── scaler.pkl
├── pca_model.pkl
├── kmeans_model.pkl
├── plots/
├── requirements.txt
└── README.md
## License

This project is released under the MIT License.
## Author

Adisa Rasak  
Independent Researcher – Materials Informatics  
Nigeria
- Melting Completion Temperature (Liquidus)

Note: The dataset is not redistributed here. Please refer to the original Kaggle source for access.
