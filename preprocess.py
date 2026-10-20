"""
Data Preprocessing Pipeline for Credit Card Fraud Detection
Handles scaling, stratified train/test splitting, and class imbalance resampling techniques.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os

def load_and_preprocess(csv_path="creditcard.csv", test_size=0.2, random_state=42):
    """
    Loads dataset, applies RobustScaler to 'Time' and 'Amount',
    and performs Stratified Train-Test Split.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop duplicates if needed, but in financial transactions duplicate values across features
    # can represent identical micro-transactions or retries. Here we retain them or log count.
    dup_count = df.duplicated().sum()
    print(f"Dataset has {dup_count} duplicate rows (retained to preserve true transaction log).")
    
    # RobustScaler is resilient to outliers in Time and Amount
    scaler = RobustScaler()
    df_scaled = df.copy()
    
    df_scaled['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df_scaled['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop original Time and Amount
    df_scaled.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Rearrange columns so features are first, Class is target
    target_col = 'Class'
    features = [c for c in df_scaled.columns if c != target_col]
    
    X = df_scaled[features]
    y = df_scaled[target_col]
    
    print(f"Features: {len(features)} ({', '.join(features[:5])}... {', '.join(features[-2:])})")
    print(f"Stratifying split (test_size={test_size}, random_state={random_state})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape[0]:,} samples (Frauds: {y_train.sum():,})")
    print(f"Test set:  {X_test.shape[0]:,} samples (Frauds: {y_test.sum():,})")
    
    return X_train, X_test, y_train, y_test, scaler, features

def get_resampled_data(X_train, y_train, method="smote", random_state=42):
    """
    Applies resampling ONLY on training data to prevent data leakage.
    Methods:
      - 'smote': Synthetic Minority Over-sampling Technique
      - 'undersample': Random Under-Sampling
      - 'none': Returns original training set
    """
    if method == "smote":
        print("Applying SMOTE oversampling to training set...")
        smote = SMOTE(sampling_strategy='auto', random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"SMOTE Train shape: {X_res.shape[0]:,} (Frauds: {y_res.sum():,}, Legit: {(y_res == 0).sum():,})")
        return X_res, y_res
        
    elif method == "undersample":
        print("Applying Random Under-Sampling to training set...")
        rus = RandomUnderSampler(sampling_strategy='auto', random_state=random_state)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        print(f"Undersampled Train shape: {X_res.shape[0]:,} (Frauds: {y_res.sum():,}, Legit: {(y_res == 0).sum():,})")
        return X_res, y_res
        
    elif method == "none":
        return X_train, y_train
        
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'smote', 'undersample', or 'none'.")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess()
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    print("Scaler saved to models/scaler.joblib")
