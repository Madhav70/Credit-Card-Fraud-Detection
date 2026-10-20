"""
Unit and Smoke Tests for Credit Card Fraud Detection Pipeline.
Used by GitHub Actions CI/CD to verify model artifacts, scaling, and predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "models"

def test_model_artifacts_exist():
    """Verify that essential model artifacts and scalers exist."""
    assert os.path.exists(os.path.join(MODEL_DIR, "best_model.joblib")), "best_model.joblib missing"
    assert os.path.exists(os.path.join(MODEL_DIR, "scaler.joblib")), "scaler.joblib missing"
    assert os.path.exists(os.path.join(MODEL_DIR, "model_comparison.json")), "model_comparison.json missing"
    assert os.path.exists(os.path.join(MODEL_DIR, "test_data.joblib")), "test_data.joblib missing"
    print("[OK] Model artifacts verified")

def test_scaler_transform():
    """Verify RobustScaler transforms amount and time properly."""
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    sample_amounts = np.array([[10.0], [100.0], [5000.0]])
    scaled = scaler.transform(sample_amounts)
    assert scaled.shape == (3, 1)
    assert not np.isnan(scaled).any()
    print("[OK] Scaler transformation verified")

def test_model_prediction_output():
    """Verify best model can predict probabilities on valid feature shapes."""
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    test_snapshot = joblib.load(os.path.join(MODEL_DIR, "test_data.joblib"))
    features = test_snapshot['features']
    
    # Create dummy single transaction
    dummy_input = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    prob = model.predict_proba(dummy_input)[0, 1]
    
    assert 0.0 <= prob <= 1.0, f"Probability {prob} out of valid [0, 1] range"
    print(f"[OK] Model inference verified (test prob: {prob:.4f})")

def test_app_syntax():
    """Verify app.py compiles cleanly."""
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()
    compile(code, "app.py", "exec")
    print("[OK] app.py syntax verified")

if __name__ == "__main__":
    test_model_artifacts_exist()
    test_scaler_transform()
    test_model_prediction_output()
    test_app_syntax()
    print("\nAll automated checks passed successfully!")
