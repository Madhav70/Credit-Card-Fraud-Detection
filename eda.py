"""
Exploratory Data Analysis (EDA) for Credit Card Fraud Detection
Analyzes creditcard.csv and generates statistical summaries and visualization charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = "eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_eda(csv_path="creditcard.csv"):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Dataset Shape and Missing Values
    print(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    null_counts = df.isnull().sum().sum()
    print(f"Total Missing Values: {null_counts}")
    
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicates:,} ({duplicates / len(df):.2%})")
    
    # 2. Class Distribution
    class_counts = df['Class'].value_counts()
    fraud_count = class_counts.get(1, 0)
    legit_count = class_counts.get(0, 0)
    fraud_ratio = (fraud_count / len(df)) * 100
    
    print("\n--- Class Breakdown ---")
    print(f"Legitimate Transactions (Class 0): {legit_count:,} ({100 - fraud_ratio:.3f}%)")
    print(f"Fraudulent Transactions (Class 1): {fraud_count:,} ({fraud_ratio:.3f}%)")
    print(f"Imbalance Ratio: 1 Fraud per {int(legit_count / fraud_count):,} Legitimate transactions")
    
    # 3. Amount Statistics
    print("\n--- Transaction Amount Statistics ---")
    legit_amt = df[df['Class'] == 0]['Amount']
    fraud_amt = df[df['Class'] == 1]['Amount']
    print(f"Legitimate - Mean: ${legit_amt.mean():.2f}, Median: ${legit_amt.median():.2f}, Max: ${legit_amt.max():.2f}")
    print(f"Fraudulent - Mean: ${fraud_amt.mean():.2f}, Median: ${fraud_amt.median():.2f}, Max: ${fraud_amt.max():.2f}")
    
    # 4. Correlation with Target Variable
    correlations = df.corr()['Class'].sort_values()
    print("\n--- Top Negative Correlations with Fraud ---")
    print(correlations.head(5).to_string())
    print("\n--- Top Positive Correlations with Fraud ---")
    print(correlations.tail(6).drop('Class', errors='ignore').to_string())
    
    # ---------------- VISUALIZATIONS ----------------
    
    # Plot 1: Class Distribution
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#1f77b4', '#d62728']
    sns.countplot(x='Class', data=df, palette=colors, ax=ax[0], hue='Class', legend=False)
    ax[0].set_yscale('log')
    ax[0].set_title("Transaction Count by Class (Log Scale)", fontsize=13, fontweight='bold', pad=10)
    ax[0].set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'], fontsize=11)
    ax[0].set_ylabel("Count (log scale)", fontsize=11)
    ax[0].set_xlabel("")
    for p in ax[0].patches:
        height = p.get_height()
        ax[0].annotate(f'{int(height):,}',
                       (p.get_x() + p.get_width() / 2., height),
                       ha='center', va='bottom', fontsize=10, xytext=(0, 5),
                       textcoords='offset points')
                       
    ax[1].pie([legit_count, fraud_count], labels=['Legitimate\n(99.83%)', 'Fraud\n(0.17%)'],
              autopct='%1.2f%%', colors=colors, startangle=45, explode=(0, 0.2),
              wedgeprops=dict(width=0.6, edgecolor='w', linewidth=2))
    ax[1].set_title("Class Imbalance Ratio", fontsize=13, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    fig.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved {plot1_path}")
    
    # Plot 2: Amount Distribution (Fraud vs Legit)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(legit_amt, bins=50, kde=True, color='#1f77b4', ax=ax[0])
    ax[0].set_title("Legitimate Transaction Amount Distribution", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("Amount ($)")
    ax[0].set_xlim([0, 1000])  # focus on primary range
    
    sns.histplot(fraud_amt, bins=50, kde=True, color='#d62728', ax=ax[1])
    ax[1].set_title("Fraudulent Transaction Amount Distribution", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("Amount ($)")
    ax[1].set_xlim([0, 1000])
    
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, "amount_distribution.png")
    fig.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved {plot2_path}")

    # Plot 3: Time vs Transaction Count and Hour of Day
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df[df['Class'] == 0]['Time'] / 3600, bins=48, color='#1f77b4', label='Legit', ax=ax[0], stat='density', common_norm=False)
    sns.histplot(df[df['Class'] == 1]['Time'] / 3600, bins=48, color='#d62728', label='Fraud', ax=ax[0], stat='density', common_norm=False)
    ax[0].set_title("Transaction Density over Time (Hours)", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("Time (Hours since start)")
    ax[0].set_ylabel("Density")
    ax[0].legend()
    
    # Amount Boxplots
    df_box = df.copy()
    df_box['Class_Label'] = df_box['Class'].map({0: 'Legit', 1: 'Fraud'})
    sns.boxplot(x='Class_Label', y='Amount', data=df_box, palette=['#1f77b4', '#d62728'], ax=ax[1], showfliers=False, hue='Class_Label', legend=False)
    ax[1].set_title("Transaction Amount Boxplot (Excluding Extreme Outliers)", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("")
    ax[1].set_ylabel("Amount ($)")
    
    plt.tight_layout()
    plot3_path = os.path.join(OUTPUT_DIR, "time_amount_comparison.png")
    fig.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved {plot3_path}")

    # Plot 4: Top Features Correlation with Class
    top_corr = pd.concat([correlations.head(6), correlations.tail(7).drop('Class', errors='ignore')])
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_corr = ['#d62728' if x < 0 else '#1f77b4' for x in top_corr.values]
    top_corr.plot(kind='barh', color=colors_corr, ax=ax)
    ax.set_title("Top Positively & Negatively Correlated Features with Fraud (Class)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plot4_path = os.path.join(OUTPUT_DIR, "top_correlations.png")
    fig.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved {plot4_path}")

    # Plot 5: Distribution of Top 4 Predictive Features (V17, V14, V12, V4)
    top_features = ['V17', 'V14', 'V12', 'V4']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, feature in enumerate(top_features):
        sns.kdeplot(df[df['Class'] == 0][feature], label='Legit', color='#1f77b4', ax=axes[i], fill=True, alpha=0.3)
        sns.kdeplot(df[df['Class'] == 1][feature], label='Fraud', color='#d62728', ax=axes[i], fill=True, alpha=0.4)
        axes[i].set_title(f"Distribution of {feature} by Class", fontsize=12, fontweight='bold')
        axes[i].legend()
    plt.tight_layout()
    plot5_path = os.path.join(OUTPUT_DIR, "top_features_distribution.png")
    fig.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"Saved {plot5_path}")

    print("\nEDA Completed Successfully! All plots saved in eda_plots/.")

if __name__ == "__main__":
    run_eda()
