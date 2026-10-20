# 💳 Credit Card Fraud Detection System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade Machine Learning solution for detecting fraudulent credit card transactions in heavily imbalanced financial data.

---

## 🌐 Live Web Application

The interactive web dashboard can be launched locally or deployed to **Streamlit Community Cloud** with 1 click:

### 1. Run Locally:
```bash
streamlit run app.py
```
*Access in browser at:* `http://localhost:8501`

### 2. Deploy Live to the Web (Free via Streamlit Cloud):
1. Sign in to [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
2. Click **New app** and select:
   - **Repository**: `Madhav70/Credit-Card-Fraud-Detection`
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. Click **Deploy!** Your interactive app will be live at a public URL (e.g., `https://credit-card-fraud-detection.streamlit.app`).

---

## 📌 Project Highlights

- **Dataset**: 284,807 European cardholder transactions, containing **492 frauds (0.173%)** — an extreme **577:1** imbalance ratio.
- **Robust Preprocessing**: Outlier-resilient `RobustScaler` on `Time` and `Amount` paired with Stratified Train/Test splitting to prevent data leakage.
- **Imbalanced Learning**: Benchmarks **Cost-Sensitive Learning** (`scale_pos_weight`, `class_weight='balanced'`) vs. **Synthetic Oversampling (SMOTE)**.
- **Evaluation Priority**: Prioritizes **Precision-Recall AUC (PR-AUC / Average Precision)**, **Recall (Sensitivity)**, and **F2-Score** over deceptive Accuracy.
- **Top Performer**: **XGBoost (`scale_pos_weight`)** achieved a **PR-AUC of 0.8596** and caught **85.7%** of frauds with only 53 false alarms out of 56,962 test transactions.
- **Interactive Web App**: A modern [Streamlit](file:///c:/Users/Admin/Downloads/Credit%20Card%20Fraud%20Detection/app.py) application featuring live transaction risk scoring, batch CSV processing, dynamic threshold tuning, and financial impact simulation.

---

## 📊 Benchmark Results (56,962 Test Transactions, 98 Frauds)

| Model | PR-AUC (Avg Precision) | ROC-AUC | Recall (Fraud Catch) | Precision | F1-Score | False Positives |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (`scale_pos_weight`)** 🏆 | **0.8596** | 0.9729 | 85.71% (84/98) | 61.31% | 0.7149 | 53 |
| **Random Forest (Balanced)** | 0.8303 | **0.9859** | 84.69% (83/98) | **72.17%** | **0.7793** | **32** |
| **Logistic Regression (SMOTE)** | 0.7235 | 0.9712 | **91.84%** (90/98) | 5.91% | 0.1110 | 1,434 |
| **Logistic Regression (Balanced)** | 0.7189 | 0.9720 | **91.84%** (90/98) | 6.09% | 0.1141 | 1,389 |

> [!TIP]
> **Why PR-AUC Matters**: With a fraud incidence of 0.173%, a naive model classifying all transactions as legitimate scores **99.83% Accuracy** and high ROC-AUC, yet catches **0 frauds**. PR-AUC directly penalizes false positive alerts while evaluating true fraud capture.

---

## 📁 Repository Structure

```
Credit Card Fraud Detection/
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
├── app.py                         # Streamlit interactive web application
├── Credit_Card_Fraud_Detection.ipynb # Complete end-to-end Jupyter Notebook
└── README.md                      # Documentation & operational guide
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis (EDA)
Generates statistical summaries and high-resolution charts in `eda_plots/`:
```bash
python eda.py
```

### 3. Train & Benchmark Models
Trains Logistic Regression, Random Forest, and XGBoost, evaluates metrics on the held-out stratified test set, and exports models to `models/`:
```bash
python train.py
```

### 4. Run Automated Tests
```bash
python test_pipeline.py
```

### 5. Launch the Interactive Web App
```bash
streamlit run app.py
```

---

## 🧠 Business Impact & Decision Threshold Simulator

In real banking operations, the default `0.50` probability threshold is rarely optimal:
$$\text{Total Expected Loss} = (\text{False Negatives} \times \text{Fraud Loss}) + (\text{False Positives} \times \text{Alert Cost})$$

- When **Fraud Loss** = **$250** and **Alert Cost** = **$15**, lowering the threshold to **~0.25–0.35** captures additional frauds, reducing overall institutional financial losses.
- The interactive Streamlit dashboard includes a **live simulator slider** allowing risk managers to dynamically tune this balance.
