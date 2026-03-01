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
    <p>Compare all 4 models — confusion matrices, ROC curves, feature importance, and business impact.</p>
</div>
""", unsafe_allow_html=True)

# ── Load ───────────────────────────────────────────────────────────────────────
log, dt, rf, gb, minmax = load_saved_models()
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
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42, stratify=Y)

with st.sidebar:
    st.markdown("### Settings")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05)

# Evaluate all 4 models
all_models = [
    ("Logistic Regression", log),
    ("Decision Tree",       dt),
    ("Random Forest",       rf),
    ("Gradient Boosting",   gb),
]
results = {name: evaluate_model(m, x_train, y_train, x_test, y_test, threshold)
           for name, m in all_models}

FIG_BG = '#ffffff'
COLORS = ['#4361ee', '#f72585', '#00d296', '#f4a261']

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([" Model Comparison ", " Charts ", " Business Impact "])

# ─── Tab 1 ────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Metric Comparison</div>', unsafe_allow_html=True)

    metrics = ["Accuracy", "CV Mean F1", "Precision", "Recall", "F1 Score"]
    comp_data = {"Metric": metrics}
    for name, res in results.items():
        comp_data[name] = [res[m] for m in metrics]

    comp_df = pd.DataFrame(comp_data)
    fmt = {name: "{:.4f}" for name in results}
    st.dataframe(comp_df.style.format(fmt), width="stretch", height=220)

    # Best model by F1
    best = max(results, key=lambda n: results[n]["F1 Score"])
    st.success(f"**{best}** achieves the highest F1 Score — {results[best]['F1 Score']:.4f}")

    st.markdown('<div class="section-title">Individual Results</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for i, (name, res) in enumerate(results.items()):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"**{name}**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{res['Accuracy']:.3f}")
            m2.metric("Precision", f"{res['Precision']:.3f}")
            m3.metric("Recall",    f"{res['Recall']:.3f}")
            m4.metric("F1",        f"{res['F1 Score']:.3f}")

# ─── Tab 2 ────────────────────────────────────────────────────────────────────
with tab2:
    # ROC curves - all 4 on one chart
    st.markdown('<div class="section-title">ROC Curves</div>', unsafe_allow_html=True)
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5), facecolor=FIG_BG)
    ax_roc.set_facecolor(FIG_BG)
    for (name, res), color in zip(results.items(), COLORS):
        ax_roc.plot(res['FPR'], res['TPR'], color=color,
                    label=f"{name}  (AUC = {res['AUC']:.3f})", linewidth=2.2)
    ax_roc.plot([0, 1], [0, 1], color='#cbd5e1', linestyle='--', linewidth=1)
    ax_roc.set_xlabel('False Positive Rate', fontsize=9)
    ax_roc.set_ylabel('True Positive Rate', fontsize=9)
    ax_roc.legend(loc='lower right', fontsize=9)
    for spine in ax_roc.spines.values(): spine.set_color('#e2e8f4')
    plt.tight_layout()
    st.pyplot(fig_roc)

    # Confusion matrices - 2x2 grid
    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    fig_cm, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor=FIG_BG)
    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res['Confusion Matrix'], annot=True, fmt='d',
                    cmap='Blues', ax=ax, cbar=False, linewidths=1, linecolor='white')
        ax.set_title(name, fontsize=9, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=8); ax.set_ylabel('Actual', fontsize=8)
        ax.set_facecolor(FIG_BG)
    plt.tight_layout()
    st.pyplot(fig_cm)

    # Feature importance - RF and GB side by side
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    fi_col1, fi_col2 = st.columns(2)

    for col, model, name in [(fi_col1, rf, "Random Forest"), (fi_col2, gb, "Gradient Boosting")]:
        with col:
            try:
                feat_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                fig_fi, ax_fi = plt.subplots(figsize=(5, 4), facecolor=FIG_BG)
                ax_fi.set_facecolor(FIG_BG)
                sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax_fi, color='#4361ee')
                ax_fi.set_title(name, fontweight='bold', fontsize=10)
                for spine in ax_fi.spines.values(): spine.set_color('#e2e8f4')
                plt.tight_layout()
                st.pyplot(fig_fi)
            except Exception:
                st.warning(f"Could not extract feature importance for {name}.")

# ─── Tab 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("Translate ML metrics into real bottom-line revenue figures for the bank.")

    colA, colB = st.columns(2)
    with colA:
        cost_churn = st.number_input("Cost of 1 Churned Customer ($)", 100.0, value=1000.0, step=100.0)
    with colB:
        cost_campaign = st.number_input("Cost of Retention Campaign ($)", 10.0, value=50.0, step=10.0)

    def impact(cm):
        tp, fp = int(cm[1][1]), int(cm[0][1])
        rev  = tp * cost_churn
        cost = (tp + fp) * cost_campaign
        return tp, rev, cost, rev - cost

    cols = st.columns(4)
    for col, (name, res) in zip(cols, results.items()):
        tp, rev, cost, net = impact(res['Confusion Matrix'])
        with col:
            st.markdown(f'<div class="impact-card"><h4>{name}</h4>', unsafe_allow_html=True)
            st.metric("Customers Saved", f"{tp}")
            st.metric("Revenue Saved",   f"${rev:,.0f}")
            st.metric("Campaign Cost",   f"-${cost:,.0f}")
            if net > 0:
                st.success(f"**Net: ${net:,.0f}**")
            else:
                st.error(f"**Loss: ${abs(net):,.0f}**")
            st.markdown('</div>', unsafe_allow_html=True)

    st.info(f"Based on {len(y_test)} test customers. Full deployment across **{len(raw_df):,}** customers scales proportionally.")
