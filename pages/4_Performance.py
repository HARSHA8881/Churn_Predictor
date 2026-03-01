import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data_utils import load_data, clean_data, encode_features, scale_features
from src.model import load_saved_models, evaluate_model
from src.styles import inject_css
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Performance | Churn Predictor", page_icon="P", layout="wide")
inject_css()

st.markdown("""
<div class="page-header">
    <h1>Performance</h1>
    <p>Compare models, view confusion matrices, ROC curves, feature importance, and business impact.</p>
</div>
""", unsafe_allow_html=True)

log, dt, minmax = load_saved_models()
if log is None:
    st.warning("No trained models found. Please go to Model Training first."); st.stop()

data_dir = os.path.join(ROOT, "data")
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")] if os.path.exists(data_dir) else []
if not csv_files:
    st.warning("No dataset found. Please upload a CSV in Dataset Explorer first."); st.stop()

with open(os.path.join(data_dir, csv_files[0]), "rb") as f:
    raw_df, df = load_data(f)

X, Y, num_cols, cat_cols = clean_data(df)
X, encoders = encode_features(X, cat_cols)
X, _ = scale_features(X, num_cols)
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=0)

with st.sidebar:
    st.markdown("### Settings")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05)

lr_res = evaluate_model(log, x_train, y_train, x_test, y_test, threshold)
dt_res = evaluate_model(dt,  x_train, y_train, x_test, y_test, threshold)

FIG_BG = '#ffffff'

tab1, tab2, tab3 = st.tabs(["Model Comparison", "Charts", "Business Impact"])

# ─── Tab 1 ────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Metric Comparison</div>', unsafe_allow_html=True)

    metrics = ["Accuracy", "CV Mean Accuracy", "Precision", "Recall", "F1 Score"]
    comp_df = pd.DataFrame({
        "Metric": metrics,
        "Logistic Regression": [lr_res[m] for m in metrics],
        "Decision Tree":       [dt_res[m] for m in metrics],
    })
    st.dataframe(
        comp_df.style.format({"Logistic Regression": "{:.4f}", "Decision Tree": "{:.4f}"}),
        use_container_width=True,
        height=220,
    )

    lr_f1, dt_f1 = lr_res['F1 Score'], dt_res['F1 Score']
    if lr_f1 > dt_f1:
        st.success("Logistic Regression wins on F1 Score — better balance of Precision and Recall.")
    elif dt_f1 > lr_f1:
        st.success("Decision Tree wins on F1 Score — handles non-linear boundaries effectively.")
    else:
        st.info("Both models are tied on F1 Score.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Logistic Regression**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{lr_res['Accuracy']:.3f}")
        c2.metric("Precision", f"{lr_res['Precision']:.3f}")
        c3.metric("Recall",    f"{lr_res['Recall']:.3f}")
        c4.metric("F1",        f"{lr_res['F1 Score']:.3f}")
    with col2:
        st.markdown("**Decision Tree**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{dt_res['Accuracy']:.3f}")
        c2.metric("Precision", f"{dt_res['Precision']:.3f}")
        c3.metric("Recall",    f"{dt_res['Recall']:.3f}")
        c4.metric("F1",        f"{dt_res['F1 Score']:.3f}")

# ─── Tab 2 ────────────────────────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
        fig_cm, axes_cm = plt.subplots(1, 2, figsize=(9, 4), facecolor=FIG_BG)
        for ax, res, title in zip(axes_cm, [lr_res, dt_res], ['Logistic Regression', 'Decision Tree']):
            sns.heatmap(res['Confusion Matrix'], annot=True, fmt='d',
                        cmap='Blues', ax=ax, cbar=False,
                        linewidths=1, linecolor='white')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('Actual', fontsize=9)
            ax.set_facecolor(FIG_BG)
        plt.tight_layout()
        st.pyplot(fig_cm)

    with col2:
        st.markdown('<div class="section-title">ROC Curves</div>', unsafe_allow_html=True)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4), facecolor=FIG_BG)
        ax_roc.set_facecolor(FIG_BG)
        ax_roc.plot(lr_res['FPR'], lr_res['TPR'], color='#4361ee',
                    label=f"LR  (AUC = {lr_res['AUC']:.3f})", linewidth=2.5)
        ax_roc.plot(dt_res['FPR'], dt_res['TPR'], color='#f72585',
                    label=f"DT  (AUC = {dt_res['AUC']:.3f})", linewidth=2.5)
        ax_roc.plot([0, 1], [0, 1], color='#cbd5e1', linestyle='--', linewidth=1)
        ax_roc.set_xlabel('False Positive Rate', fontsize=9)
        ax_roc.set_ylabel('True Positive Rate', fontsize=9)
        ax_roc.legend(loc='lower right', fontsize=9)
        for spine in ax_roc.spines.values(): spine.set_color('#e2e8f4')
        plt.tight_layout()
        st.pyplot(fig_roc)

    st.markdown('<div class="section-title">Feature Importance (Decision Tree)</div>', unsafe_allow_html=True)
    try:
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': dt.feature_importances_}
                               ).sort_values('Importance', ascending=False)
        fig_imp, ax_imp = plt.subplots(figsize=(10, 4), facecolor=FIG_BG)
        ax_imp.set_facecolor(FIG_BG)
        sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax_imp, color='#4361ee')
        ax_imp.set_title('Decision Tree Feature Importance', fontweight='bold', fontsize=11)
        for spine in ax_imp.spines.values(): spine.set_color('#e2e8f4')
        plt.tight_layout()
        st.pyplot(fig_imp)
    except Exception:
        st.warning("Could not extract feature importance.")

# ─── Tab 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("Translate ML metrics (True Positives / False Positives) into real bottom-line revenue figures.")

    colA, colB = st.columns(2)
    with colA:
        cost_churn = st.number_input("Cost of 1 Churned Customer ($)", 100.0, value=1000.0, step=100.0)
    with colB:
        cost_campaign = st.number_input("Cost of Retention Campaign ($)", 10.0, value=50.0, step=10.0)

    def impact(cm):
        tp, fp = cm[1][1], cm[0][1]
        rev  = tp * cost_churn
        cost = (tp + fp) * cost_campaign
        return tp, rev, cost, rev - cost

    c1, c2 = st.columns(2)
    for col, cm, lbl in [(c1, lr_res['Confusion Matrix'], "Logistic Regression"),
                         (c2, dt_res['Confusion Matrix'], "Decision Tree")]:
        tp, rev, cost, net = impact(cm)
        with col:
            st.markdown(f'<div class="impact-card"><h4>{lbl}</h4>', unsafe_allow_html=True)
            st.metric("Customers Saved (TP)", f"{tp}")
            st.metric("Revenue Saved",        f"${rev:,.2f}")
            st.metric("Campaign Cost",        f"-${cost:,.2f}")
            if net > 0:
                st.success(f"**Net Profit: ${net:,.2f}**")
            else:
                st.error(f"**Net Loss: ${abs(net):,.2f}**")
            st.markdown('</div>', unsafe_allow_html=True)

    st.info(f"Based on {len(y_test)} test customers. Full deployment across **{len(raw_df):,}** customers scales proportionally.")
