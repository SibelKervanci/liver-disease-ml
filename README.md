# An Interpretable Ensemble ML Pipeline for Dual-Task Liver Disease Prediction: Prognosis and Diagnosis

This repository contains the source code for a leakage-free, interpretable machine learning pipeline that addresses two clinical tasks at once: **survival prognosis** in cirrhosis patients and **diagnostic classification** of liver disease. The pipeline combines strict fold-confined preprocessing, ensemble learning, Bayesian hyperparameter optimization, dual-level interpretability (SHAP + LIME), and an explicit data-leakage ablation that quantifies the accuracy inflation behind many optimistic results reported in the literature.

> **Note:** This is the companion code for a manuscript currently under review. Publication details (journal, volume, DOI) will be added here once the paper is accepted.

## Overview

Two independent, real-world clinical cohorts are used:

| Dataset | Task | Records | Source |
|---|---|---|---|
| Mayo Clinic Primary Biliary Cirrhosis | Survival prognosis (DS1) | 418 | UCI |
| Indian Liver Patient Dataset (ILPD) | Diagnostic classification (DS2) | 583 | UCI |

The pipeline trains seven base classifiers (KNN, SVC, MLP, Random Forest, XGBoost, LightGBM, TabNet) and consolidates them through four ensemble strategies (soft voting, stacking, blending, and a custom dynamic ensemble). All preprocessing — imputation, scaling, SMOTE oversampling, and feature selection — is confined to the training folds to prevent data leakage.

## Key Features

- **Leakage-free pipeline:** all preprocessing is performed inside cross-validation folds.
- **Dual-task design:** prognosis and diagnosis handled in one unified framework.
- **Bayesian hyperparameter optimization:** Optuna with a Tree-structured Parzen Estimator (TPE) sampler.
- **Comprehensive evaluation:** accuracy, precision, recall, F1, MCC, ROC-AUC, PR-AUC, and Brier score, with five-fold cross-validation and Friedman / Wilcoxon significance tests.
- **Interpretability:** global SHAP feature attributions and local LIME explanations.
- **Data-leakage ablation:** a controlled experiment showing how pre-split oversampling and removed regularization inflate standalone-model accuracy.
- **Subgroup fairness:** performance reported across sex, age, and biomarker strata.

## Repository Structure

```
liver-disease-ml/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── cirrhosis_ILDP_bayesian_pipeline.py  # Main pipeline script
└── data/                              # Place datasets here (see Data section)
```

## Installation

This project requires **Python 3.9+**. Clone the repository and install the dependencies:

```bash
git clone https://github.com/SibelKervanci/liver-disease-ml.git
cd liver-disease-ml
pip install -r requirements.txt
```

Using a virtual environment is recommended:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The datasets are publicly available from the UCI Machine Learning Repository and are **not redistributed** in this repository. Download them and place them in the `data/` folder before running the pipeline.

- **Cirrhosis Patient Survival dataset:** https://doi.org/10.24432/C5R02C
- **Indian Liver Patient Dataset (ILPD):** https://doi.org/10.24432/C5D02C

The ILPD file is expected as a semicolon-separated CSV named `ILPD.csv` with the following columns:

```
Age;Gender;TB;DB;Alkphos;Sgpt;Sgot;TP;ALB;A/G_Ratio;Selector
```

## Usage

Run the full pipeline with:

```bash
python cirrhosis_ILDP_bayesian_pipeline.py
```

The script trains all base classifiers and ensembles on both cohorts, performs cross-validation and statistical testing, runs the data-leakage ablation, and generates the SHAP and LIME interpretability outputs. Result tables and figures are written to the output directory created at runtime.

## Results Summary

The standard Blending ensemble achieved the best overall performance on both tasks:

| Cohort | Best Model | Accuracy | F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|
| Cirrhosis (DS1) | Blending | 0.8452 | 0.7797 | 0.6674 | 0.9038 |
| ILPD (DS2) | Blending | 0.8120 | 0.8791 | 0.5097 | 0.8373 |

The data-leakage ablation showed that, under pre-split oversampling with regularization removed, standalone models inflate sharply (KNN +27.00% to 0.9281; Random Forest +18.11% to 0.9162 on ILPD), while the Blending ensemble remained the most resistant to this inflation.

## Citation

If you use this code, please cite the associated paper. The full citation will be added here once the manuscript is published. In the meantime, you can reference this repository directly.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

İlkay Sibel Kervancı — Computer Engineering Department, Gaziantep University
Email: skervanci@gantep.edu.tr
