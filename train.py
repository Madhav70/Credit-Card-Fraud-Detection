"""
Model Training, Benchmarking, and Evaluation Pipeline for Credit Card Fraud Detection.
Trains Logistic Regression, Random Forest, and XGBoost with class balancing,
benchmarks them on PR-AUC, ROC-AUC, Recall, and saves the best model and metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)
from preprocess import load_and_preprocess, get_resampled_data

MODEL_DIR = "models"
EVAL_DIR = "evaluation_plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    """
    Evaluates model probabilities and predictions at given threshold.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        
    y_pred = (y_prob >= threshold).astype(int)
    
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_test, y_pred, beta=2, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    return {
        "model_name": name,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "probabilities": y_prob
    }

def train_and_benchmark():
    # 1. Load and Preprocess Data
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess()
    
    # Save test dataset snapshot (features + true labels) for quick interactive Streamlit testing
    test_snapshot = {
        'X_test': X_test.copy(),
        'y_test': y_test.copy(),
        'scaler': scaler,
        'features': features
    }
    joblib.dump(test_snapshot, os.path.join(MODEL_DIR, "test_data.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"Calculated XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 2. Define Models
    models_to_train = {
        "Logistic Regression (Balanced)": {
            "model": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            "data_type": "original"
        },
        "Random Forest (Balanced)": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced',
                                           random_state=42, n_jobs=-1),
            "data_type": "original"
        },
        "XGBoost (scale_pos_weight)": {
            "model": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08,
                                 scale_pos_weight=scale_pos_weight, random_state=42,
                                 eval_metric='logloss', n_jobs=-1),
            "data_type": "original"
        },
        "Logistic Regression (SMOTE)": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "data_type": "smote"
        }
    }
    
    # Prepare SMOTE dataset once
    X_train_smote, y_train_smote = get_resampled_data(X_train, y_train, method="smote")
    
    results = {}
    fitted_models = {}
    probs_dict = {}
    
    for name, config in models_to_train.items():
        print(f"\n==========================================")
        print(f"Training: {name}")
        print(f"==========================================")
        clf = config["model"]
        
        if config["data_type"] == "smote":
            clf.fit(X_train_smote, y_train_smote)
        else:
            clf.fit(X_train, y_train)
            
        fitted_models[name] = clf
        
        # Evaluate
        eval_metrics = evaluate_model(name, clf, X_test, y_test)
        probs_dict[name] = eval_metrics.pop("probabilities")
        results[name] = eval_metrics
        
        print(f"Results for {name}:")
        print(f"  PR-AUC (Avg Precision): {eval_metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC:               {eval_metrics['roc_auc']:.4f}")
        print(f"  Recall (Fraud Catch):  {eval_metrics['recall']:.4f} ({eval_metrics['true_positives']}/{eval_metrics['true_positives'] + eval_metrics['false_negatives']})")
        print(f"  Precision:             {eval_metrics['precision']:.4f}")
        print(f"  F1-Score:              {eval_metrics['f1']:.4f}")
        print(f"  False Positives:       {eval_metrics['false_positives']}")
        
    # 3. Determine Best Model (Based on PR-AUC and F1)
    # PR-AUC is the gold standard for fraud detection
    best_model_name = max(results, key=lambda k: (results[k]['pr_auc'], results[k]['f1']))
    print(f"\n**************************************************")
    print(f"BEST PERFORMING MODEL: {best_model_name}")
    print(f"PR-AUC: {results[best_model_name]['pr_auc']}, Recall: {results[best_model_name]['recall']}")
    print(f"**************************************************")
    
    # Save models
    joblib.dump(fitted_models[best_model_name], os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump(fitted_models["XGBoost (scale_pos_weight)"], os.path.join(MODEL_DIR, "xgb_model.joblib"))
    joblib.dump(fitted_models["Random Forest (Balanced)"], os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(fitted_models["Logistic Regression (Balanced)"], os.path.join(MODEL_DIR, "lr_model.joblib"))
    
    # Save comparison metrics
    summary_to_save = {
        "best_model_name": best_model_name,
        "models": results
    }
    with open(os.path.join(MODEL_DIR, "model_comparison.json"), "w") as f:
        json.dump(summary_to_save, f, indent=4)
        
    # 4. Generate Benchmarking Curves & Plots
    # Plot 1: Precision-Recall Curves
    plt.figure(figsize=(9, 6))
    for name, probs in probs_dict.items():
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
        pr_score = results[name]['pr_auc']
        plt.plot(recall_vals, precision_vals, lw=2, label=f"{name} (PR-AUC = {pr_score:.3f})")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves (Fraud Detection)", fontsize=13, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    pr_curve_path = os.path.join(EVAL_DIR, "pr_curves.png")
    plt.savefig(pr_curve_path, dpi=300)
    plt.close()
    
    # Plot 2: ROC Curves
    plt.figure(figsize=(9, 6))
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_score = results[name]['roc_auc']
        plt.plot(fpr, tpr, lw=2, label=f"{name} (ROC-AUC = {roc_score:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Random Guess (0.50)")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    roc_curve_path = os.path.join(EVAL_DIR, "roc_curves.png")
    plt.savefig(roc_curve_path, dpi=300)
    plt.close()
    
    # Plot 3: Confusion Matrices Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (name, metrics) in enumerate(results.items()):
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Pred Legit', 'Pred Fraud'],
                    yticklabels=['Actual Legit', 'Actual Fraud'])
        axes[idx].set_title(f"{name}\nPR-AUC: {metrics['pr_auc']:.3f} | Recall: {metrics['recall']:.3f}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(EVAL_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # Plot 4: Feature Importance of Best Tree Model (XGBoost or RF)
    tree_model = fitted_models.get("XGBoost (scale_pos_weight)") or fitted_models.get("Random Forest (Balanced)")
    if hasattr(tree_model, "feature_importances_"):
        plt.figure(figsize=(10, 7))
        importances = pd.Series(tree_model.feature_importances_, index=features).sort_values(ascending=True)
        importances.tail(15).plot(kind='barh', color='#1f77b4')
        plt.title("Top 15 Most Important Features (XGBoost / Tree Model)", fontsize=13, fontweight='bold')
        plt.xlabel("Relative Feature Importance")
        plt.tight_layout()
        fi_path = os.path.join(EVAL_DIR, "feature_importance.png")
        plt.savefig(fi_path, dpi=300)
        plt.close()
        
    print(f"\nAll models trained and saved to {MODEL_DIR}/.")
    print(f"Evaluation charts saved to {EVAL_DIR}/.")

if __name__ == "__main__":
    train_and_benchmark()
