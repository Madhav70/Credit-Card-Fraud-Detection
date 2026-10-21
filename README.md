# Credit Card Fraud Detection System
**Live Web Application (GitHub Pages)**: [https://madhav70.github.io/Credit-Card-Fraud-Detection/](https://madhav70.github.io/Credit-Card-Fraud-Detection/)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?style=for-the-badge&logo=github)](https://madhav70.github.io/Credit-Card-Fraud-Detection/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade Machine Learning solution for detecting fraudulent credit card transactions in heavily imbalanced financial data.

---

## Live Interactive Application

The full interactive dashboard is hosted directly on **GitHub Pages**:

**[https://madhav70.github.io/Credit-Card-Fraud-Detection/](https://madhav70.github.io/Credit-Card-Fraud-Detection/)**

Features available on the live page:
1. **Executive KPI Dashboard**: Overview of 284,807 transactions, 0.173% fraud rate, and model metrics.
2. **Interactive EDA Charts**: Class imbalance breakdowns, amount profiles, and correlation rankings.
3. **Dynamic Decision Threshold Simulator**: Adjust the classification cutoff (0.01 to 0.99) with live recalculations of Recall, False Positives, and financial loss ($FN \times \$250 + FP \times \$15$).
4. **Real-time Transaction Risk Evaluator**: Input any amount, time, and PCA coordinates or test pre-loaded normal vs. stolen card presets with live risk gauge.
5. **Batch Audit Stream**: Simulated stream of incoming transactions with automated risk classification.

---

## Project Highlights

- **Dataset**: 284,807 European cardholder transactions, containing **492 frauds (0.173%)** — an extreme **577:1** imbalance ratio.
- **Robust Preprocessing**: Outlier-resilient `RobustScaler` on `Time` and `Amount` paired with Stratified Train/Test splitting to prevent data leakage.
- **Imbalanced Learning**: Benchmarks **Cost-Sensitive Learning** (`scale_pos_weight`, `class_weight='balanced'`) vs. **Synthetic Oversampling (SMOTE)**.
- **Evaluation Priority**: Prioritizes **Precision-Recall AUC (PR-AUC / Average Precision)**, **Recall (Sensitivity)**, and **F2-Score** over deceptive Accuracy.
- **Top Performer**: **XGBoost (`scale_pos_weight`)** achieved a **PR-AUC of 0.8596** and caught **85.7%** of frauds with only 53 false alarms out of 56,962 test transactions.
- **Streamlit Local Application**: Also includes a full Python [`app.py`](file:///c:/Users/Admin/Downloads/Credit%20Card%20Fraud%20Detection/app.py) for local testing or custom CSV uploads.

---

## Benchmark Results (56,962 Test Transactions, 98 Frauds)

| Model | PR-AUC (Avg Precision) | ROC-AUC | Recall (Fraud Catch) | Precision | F1-Score | False Positives |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (`scale_pos_weight`)** | **0.8596** | 0.9729 | 85.71% (84/98) | 61.31% | 0.7149 | 53 |
| **Random Forest (Balanced)** | 0.8303 | **0.9859** | 84.69% (83/98) | **72.17%** | **0.7793** | **32** |
| **Logistic Regression (SMOTE)** | 0.7235 | 0.9712 | **91.84%** (90/98) | 5.91% | 0.1110 | 1,434 |
| **Logistic Regression (Balanced)** | 0.7189 | 0.9720 | **91.84%** (90/98) | 6.09% | 0.1141 | 1,389 |

> [!TIP]
> **Why PR-AUC Matters**: With a fraud incidence of 0.173%, a naive model classifying all transactions as legitimate scores **99.83% Accuracy** and high ROC-AUC, yet catches **0 frauds**. PR-AUC directly penalizes false positive alerts while evaluating true fraud capture.

---

## Repository Structure

```
Credit Card Fraud Detection/
├── index.html                     # Live interactive web application (GitHub Pages)
├── creditcard.csv                 # Original Kaggle dataset (284,807 rows)
├── requirements.txt               # Project dependencies
├── test_pipeline.py               # Automated smoke & unit tests
├── eda.py                         # Standalone Exploratory Data Analysis & visual generation script
├── eda_plots/                     # Generated high-resolution EDA figures
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── time_amount_comparison.png
│   ├── top_correlations.png
│   └── top_features_distribution.png
├── preprocess.py                  # Robust scaling, train-test stratified split, SMOTE logic
├── train.py                       # Model training, benchmarking, PR/ROC evaluation & artifact export
├── models/                        # Serialized models and test data
│   ├── best_model.joblib          # Best model (XGBoost)
│   ├── xgb_model.joblib
│   ├── rf_model.joblib
│   ├── lr_model.joblib
│   ├── scaler.joblib              # Fitted RobustScaler
│   ├── test_data.joblib           # Preprocessed test partition for instant inference
│   └── model_comparison.json     # Benchmark performance metrics
├── evaluation_plots/              # Benchmark curves and diagnostic charts
│   ├── pr_curves.png              # Precision-Recall curves across models
│   ├── roc_curves.png             # Receiver Operating Characteristic curves
│   ├── confusion_matrices.png     # 2x2 comparison grid of confusion matrices
│   └── feature_importance.png     # Top predictive features (XGBoost Gain)
├── app.py                         # Streamlit Python web dashboard
├── Credit_Card_Fraud_Detection.ipynb # Complete end-to-end Jupyter Notebook
└── README.md                      # Documentation & operational guide
```

---

## Getting Started Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis (EDA)
```bash
python eda.py
```

### 3. Train & Benchmark Models
```bash
python train.py
```

### 4. Run Automated Smoke Tests
```bash
python test_pipeline.py
```

### 5. Launch Streamlit Python Web App
```bash
streamlit run app.py
```
