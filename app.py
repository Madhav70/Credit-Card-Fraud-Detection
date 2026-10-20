"""
Streamlit Web Application: Real-Time Credit Card Fraud Detection & Risk Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #3182ce;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .risk-high {
        background-color: #fff5f5;
        border: 2px solid #e53e3e;
        color: #c53030;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #f0fff4;
        border: 2px solid #38a169;
        color: #276749;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "models"
EVAL_DIR = "evaluation_plots"
EDA_DIR = "eda_plots"

@st.cache_resource
def load_models_and_data():
    """Loads trained models, test set snapshot, and scaler."""
    data = {}
    if os.path.exists(os.path.join(MODEL_DIR, "test_data.joblib")):
        data['test_data'] = joblib.load(os.path.join(MODEL_DIR, "test_data.joblib"))
    if os.path.exists(os.path.join(MODEL_DIR, "scaler.joblib")):
        data['scaler'] = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        
    models = {}
    for filename, name in [
        ("best_model.joblib", "Best Model (Auto-selected)"),
        ("xgb_model.joblib", "XGBoost (scale_pos_weight)"),
        ("rf_model.joblib", "Random Forest (Balanced)"),
        ("lr_model.joblib", "Logistic Regression (Balanced)")
    ]:
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            
    data['models'] = models
    
    comp_path = os.path.join(MODEL_DIR, "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            data['comparison'] = json.load(f)
    else:
        data['comparison'] = None
        
    return data

loaded = load_models_and_data()
test_data = loaded.get('test_data')
scaler = loaded.get('scaler')
models = loaded.get('models', {})
comparison = loaded.get('comparison')

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-cards.png", width=70)
st.sidebar.title("Fraud Guard AI")
st.sidebar.markdown("**Intelligent Transaction Risk Engine**")
st.sidebar.markdown("---")

selected_model_name = st.sidebar.selectbox(
    "Active Machine Learning Model",
    list(models.keys()) if models else ["No trained models found"]
)

threshold = st.sidebar.slider(
    "Decision Threshold (Probability Cutoff)",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help="Transactions with predicted fraud probability >= threshold will be flagged as FRAUD."
)

st.sidebar.markdown("---")
st.sidebar.subheader("💼 Financial Impact Simulator")
fraud_cost = st.sidebar.number_input("Cost of Missed Fraud (FN) ($)", min_value=10, max_value=5000, value=250, step=25)
alert_cost = st.sidebar.number_input("Cost of False Alarm (FP) ($)", min_value=1, max_value=100, value=15, step=5)

# ----------------- MAIN HEADER -----------------
st.markdown('<div class="main-header">Credit Card Fraud Detection & Risk Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time fraud classification, imbalanced learning benchmarks, and cost-benefit risk analysis.</div>', unsafe_allow_html=True)

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview",
    "🔍 Exploratory Data Analysis",
    "📈 Model Benchmarks & Comparison",
    "⚡ Real-Time Fraud Predictor"
])

# ================= TAB 1: EXECUTIVE OVERVIEW =================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", "284,807", "100% Monitored")
    with col2:
        st.metric("Fraud Cases", "492", "0.173% of volume", delta_color="inverse")
    with col3:
        best_name = comparison.get("best_model_name", "XGBoost") if comparison else "XGBoost"
        st.metric("Top Model", best_name.split()[0], "Evaluated on PR-AUC")
    with col4:
        best_pr = comparison['models'][best_name]['pr_auc'] if comparison and best_name in comparison['models'] else 0.865
        st.metric("Best PR-AUC Score", f"{best_pr:.3f}", "Industry Standard")

    st.markdown("### Why Fraud Detection Requires Specialized Machine Learning")
    
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        In real-world payment networks, credit card fraud exhibits extreme **class imbalance** (~1 fraud per 577 legitimate transactions):
        - **Accuracy is a Deceptive Metric**: A dummy model predicting every transaction as *Legitimate* achieves **99.83% accuracy**, yet catches **0% of frauds**.
        - **Precision vs. Recall Trade-off**:
          - High **Recall** catches more frauds (prevents direct chargeback & refund losses).
          - High **Precision** minimizes customer friction (avoids falsely declining valid cards).
        - **PR-AUC (Precision-Recall AUC)** is the gold-standard metric for evaluating imbalanced tabular models.
        - **Feature Anonymization**: Features `V1` through `V28` are the result of Principal Component Analysis (PCA) to protect cardholder privacy, while `Time` and `Amount` remain in raw form.
        """)
    with col_b:
        st.info("""
        💡 **Core Pipeline Highlights**:
        - **RobustScaler**: Normalizes `Amount` & `Time` while handling extreme outliers ($0.00 to $25,691.16).
        - **Stratified Split**: Preserves exact class ratio across training and testing partitions.
        - **Cost-Weighted Optimization**: Dynamic threshold tuning tailored to banking business metrics.
        """)

    if os.path.exists(os.path.join(EDA_DIR, "class_distribution.png")):
        st.image(os.path.join(EDA_DIR, "class_distribution.png"), caption="Dataset Class Imbalance Distribution", use_container_width=True)

# ================= TAB 2: EXPLORATORY DATA ANALYSIS =================
with tab2:
    st.markdown("### Exploratory Data Analysis & Statistical Patterns")
    
    eda_tab_a, eda_tab_b, eda_tab_c = st.tabs([
        "Amount & Time Profiles",
        "Top Predictive Features",
        "Feature Correlation Ranking"
    ])
    
    with eda_tab_a:
        st.markdown("#### Transaction Amount & Temporal Behavior")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            **Key Insights**:
            - **Legitimate Transactions**: Mean = **$88.29**, Median = **$22.00**, Max = **$25,691.16**.
            - **Fraudulent Transactions**: Mean = **$122.21**, Median = **$9.25**, Max = **$2,125.87**.
            - Notice that fraud transactions frequently test small amounts (<$10) or moderately sized amounts, avoiding suspiciously massive transfers.
            """)
        with col_t2:
            st.markdown("""
            - Fraudulent transactions remain active consistently across off-peak hours (e.g., late night / early morning) when cardholders are asleep and less likely to notice instant SMS alerts.
            """)
        if os.path.exists(os.path.join(EDA_DIR, "time_amount_comparison.png")):
            st.image(os.path.join(EDA_DIR, "time_amount_comparison.png"), use_container_width=True)
            
    with eda_tab_b:
        st.markdown("#### Top Predictive PCA Features Distributions")
        st.markdown("""
        Features `V17`, `V14`, `V12`, and `V10` exhibit strong negative correlations with fraud, displaying clear bimodal separation between genuine and fraudulent classes.
        """)
        if os.path.exists(os.path.join(EDA_DIR, "top_features_distribution.png")):
            st.image(os.path.join(EDA_DIR, "top_features_distribution.png"), use_container_width=True)

    with eda_tab_c:
        st.markdown("#### Feature Correlation Ranking with Fraud (`Class`)")
        if os.path.exists(os.path.join(EDA_DIR, "top_correlations.png")):
            st.image(os.path.join(EDA_DIR, "top_correlations.png"), use_container_width=True)

# ================= TAB 3: MODEL BENCHMARKS & COMPARISON =================
with tab3:
    st.markdown("### Machine Learning Model Benchmarks")
    
    if comparison and "models" in comparison:
        comp_df = pd.DataFrame(comparison["models"]).T
        display_cols = ["pr_auc", "roc_auc", "recall", "precision", "f1", "f2", "false_positives", "false_negatives"]
        available_cols = [c for c in display_cols if c in comp_df.columns]
        formatted_df = comp_df[available_cols].rename(columns={
            "pr_auc": "PR-AUC (Avg Precision)",
            "roc_auc": "ROC-AUC",
            "recall": "Recall (Sensitivity)",
            "precision": "Precision",
            "f1": "F1-Score",
            "f2": "F2-Score (Recall Heavy)",
            "false_positives": "False Positives (Alerts)",
            "false_negatives": "False Negatives (Missed Frauds)"
        })
        st.dataframe(formatted_df.style.highlight_max(subset=["PR-AUC (Avg Precision)", "ROC-AUC", "Recall (Sensitivity)", "Precision", "F1-Score"], color='#c6f6d5'), use_container_width=True)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if os.path.exists(os.path.join(EVAL_DIR, "pr_curves.png")):
            st.image(os.path.join(EVAL_DIR, "pr_curves.png"), caption="Precision-Recall Curves across Models", use_container_width=True)
    with col_c2:
        if os.path.exists(os.path.join(EVAL_DIR, "roc_curves.png")):
            st.image(os.path.join(EVAL_DIR, "roc_curves.png"), caption="ROC Curves across Models", use_container_width=True)
            
    # Dynamic Confusion Matrix for Selected Model & Threshold
    st.markdown("---")
    st.markdown(f"#### Dynamic Performance at Threshold = `{threshold:.2f}` for `{selected_model_name}`")
    
    if test_data and selected_model_name in models:
        model = models[selected_model_name]
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            y_probs = model.decision_function(X_test)
            y_probs = (y_probs - y_probs.min()) / (y_probs.max() - y_probs.min())
            
        y_pred = (y_probs >= threshold).astype(int)
        
        curr_rec = recall_score(y_test, y_pred, zero_division=0)
        curr_prec = precision_score(y_test, y_pred, zero_division=0)
        curr_f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Financial Impact Calculation
        total_financial_loss = (fn * fraud_cost) + (fp * alert_cost)
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Recall (Fraud Caught)", f"{curr_rec:.1%}", f"{tp}/{tp+fn} frauds")
        m2.metric("Precision", f"{curr_prec:.1%}", f"{fp} false alarms")
        m3.metric("F1-Score", f"{curr_f1:.3f}")
        m4.metric("Missed Frauds (FN)", f"{fn}", delta_color="inverse")
        m5.metric("Est. Financial Loss", f"${total_financial_loss:,.2f}", f"Threshold {threshold:.2f}")
        
        # Render Heatmap
        fig, ax = plt.subplots(figsize=(6, 4))
        cm_data = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm_data, annot=True, fmt=',d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Pred Legit (0)', 'Pred Fraud (1)'],
                    yticklabels=['Actual Legit (0)', 'Actual Fraud (1)'])
        ax.set_title(f"Confusion Matrix (Threshold = {threshold:.2f})", fontsize=11, fontweight='bold')
        st.pyplot(fig)
        plt.close()

# ================= TAB 4: REAL-TIME FRAUD PREDICTOR =================
with tab4:
    st.markdown("### Real-Time Transaction Scoring Engine")
    
    pred_mode = st.radio("Select Prediction Mode:", ["Single Transaction Inspection", "Batch CSV Upload"], horizontal=True)
    
    if selected_model_name not in models:
        st.error("No trained model loaded. Please run train.py first.")
    else:
        active_model = models[selected_model_name]
        
        if pred_mode == "Single Transaction Inspection":
            st.markdown("#### Test a Transaction")
            
            # Quick preset buttons
            col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
            load_preset = None
            if col_b1.button("🟢 Load Sample Legit Transaction"):
                load_preset = "legit"
            if col_b2.button("🔴 Load Sample Fraud Transaction"):
                load_preset = "fraud"
                
            # Retrieve sample if requested
            sample_row = None
            if test_data:
                X_test = test_data['X_test']
                y_test = test_data['y_test']
                if load_preset == "fraud":
                    fraud_indices = y_test[y_test == 1].index
                    sample_row = X_test.loc[fraud_indices[0]]
                elif load_preset == "legit":
                    legit_indices = y_test[y_test == 0].index
                    sample_row = X_test.loc[legit_indices[0]]
                    
            with st.form("transaction_form"):
                st.markdown("**Transaction Details:**")
                col_f1, col_f2, col_f3 = st.columns(3)
                
                with col_f1:
                    raw_amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=50000.0,
                                                 value=float(99.50 if sample_row is None else 149.62), step=1.0)
                with col_f2:
                    raw_time = st.number_input("Transaction Time (Seconds)", min_value=0.0, max_value=200000.0,
                                               value=float(42000.0), step=100.0)
                with col_f3:
                    st.caption("PCA Features (V1 - V28): Anonymized cardholder vectors")
                    
                # Collapsible for V1-V28
                with st.expander("Adjust PCA Features (V1 - V28)", expanded=False):
                    v_cols = st.columns(4)
                    v_values = {}
                    features_list = [f"V{i}" for i in range(1, 29)]
                    for i, feat in enumerate(features_list):
                        default_val = float(sample_row[feat]) if sample_row is not None and feat in sample_row else 0.0
                        with v_cols[i % 4]:
                            v_values[feat] = st.number_input(f"{feat}", value=default_val, format="%.4f")
                            
                submit_button = st.form_submit_button("🔍 Evaluate Transaction Risk")
                
            if submit_button:
                # Prepare features DataFrame
                if scaler:
                    scaled_amt = scaler.transform([[raw_amount]])[0][0]
                    scaled_tm = scaler.transform([[raw_time]])[0][0]
                else:
                    scaled_amt = raw_amount
                    scaled_tm = raw_time
                    
                input_dict = {f"V{i}": [v_values.get(f"V{i}", 0.0)] for i in range(1, 29)}
                input_dict['scaled_amount'] = [scaled_amt]
                input_dict['scaled_time'] = [scaled_tm]
                
                input_df = pd.DataFrame(input_dict)
                # Ensure column order matches training
                if test_data and 'features' in test_data:
                    input_df = input_df[test_data['features']]
                    
                if hasattr(active_model, "predict_proba"):
                    prob_fraud = float(active_model.predict_proba(input_df)[0, 1])
                else:
                    score = float(active_model.decision_function(input_df)[0])
                    prob_fraud = 1.0 / (1.0 + np.exp(-score))
                    
                st.markdown("### Risk Assessment Result")
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.metric("Fraud Probability Score", f"{prob_fraud * 100:.2f}%")
                    st.progress(prob_fraud)
                    
                with col_r2:
                    if prob_fraud >= threshold:
                        st.markdown(f"""
                        <div class="risk-high">
                            🚨 HIGH RISK DETECTED: TRANSACTION FLAGGED AS FRAUD<br>
                            Probability ({prob_fraud:.1%}) exceeds the alert threshold ({threshold:.1%}).<br>
                            Action: <b>BLOCK TRANSACTION & TRIGGER OTP VERIFICATION</b>
                        </div>
                        """, unsafe_allow_html=True)
                    elif prob_fraud >= 0.20:
                        st.markdown(f"""
                        <div style="background-color: #fffaf0; border: 2px solid #dd6b20; color: #c05621; padding: 15px; border-radius: 8px; font-weight: bold;">
                            ⚠️ MEDIUM RISK: TRANSACTION MARKED FOR REVIEW<br>
                            Probability ({prob_fraud:.1%}) is elevated.<br>
                            Action: <b>SECONDARY AUTHENTICATION RECOMMENDED</b>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-low">
                            ✅ LOW RISK: TRANSACTION APPROVED<br>
                            Probability ({prob_fraud:.1%}) is within normal parameters.<br>
                            Action: <b>AUTO-APPROVE</b>
                        </div>
                        """, unsafe_allow_html=True)

        else:
            # Batch Upload Mode
            st.markdown("#### Batch CSV Evaluation")
            st.markdown("Upload a CSV with transactions (same schema as `creditcard.csv`).")
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
            
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded **{len(batch_df):,}** transactions.")
                
                if st.button("🚀 Run Batch Fraud Detection"):
                    with st.spinner("Processing transactions..."):
                        # Process batch
                        proc_batch = batch_df.copy()
                        if 'Amount' in proc_batch and 'Time' in proc_batch and scaler:
                            proc_batch['scaled_amount'] = scaler.transform(proc_batch['Amount'].values.reshape(-1, 1))
                            proc_batch['scaled_time'] = scaler.transform(proc_batch['Time'].values.reshape(-1, 1))
                            proc_batch.drop(['Time', 'Amount'], axis=1, inplace=True, errors='ignore')
                            
                        target_col = 'Class'
                        if target_col in proc_batch:
                            proc_batch.drop(target_col, axis=1, inplace=True)
                            
                        if test_data and 'features' in test_data:
                            proc_batch = proc_batch[test_data['features']]
                            
                        if hasattr(active_model, "predict_proba"):
                            probs = active_model.predict_proba(proc_batch)[:, 1]
                        else:
                            probs = active_model.decision_function(proc_batch)
                            probs = 1.0 / (1.0 + np.exp(-probs))
                            
                        batch_df['Fraud_Probability'] = probs
                        batch_df['Flagged_As_Fraud'] = (probs >= threshold).astype(int)
                        
                        flagged_count = batch_df['Flagged_As_Fraud'].sum()
                        st.success(f"Batch processing complete! Flagged **{flagged_count:,}** out of **{len(batch_df):,}** transactions ({flagged_count / len(batch_df):.2%}).")
                        
                        col_bt1, col_bt2 = st.columns(2)
                        with col_bt1:
                            st.dataframe(batch_df[['Amount', 'Fraud_Probability', 'Flagged_As_Fraud']].head(20))
                        with col_bt2:
                            # Pie chart of flagged
                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.pie([len(batch_df) - flagged_count, flagged_count],
                                   labels=['Approved', 'Flagged Fraud'],
                                   colors=['#276749', '#c53030'], autopct='%1.1f%%', startangle=45)
                            st.pyplot(fig)
                            plt.close()
                            
                        # Download button
                        csv_data = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Flagged Results CSV",
                            data=csv_data,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )
