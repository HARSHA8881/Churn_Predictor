import streamlit as st
import pandas as pd
import numpy as np
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data_utils import load_data, clean_data, encode_features, scale_features
from src.model import load_saved_models
from src.styles import inject_css

st.set_page_config(page_title="Churn Predictor | Churn Predictor", page_icon="R", layout="wide")
inject_css()

st.markdown("""
<div class="page-header">
    <h1>Churn Predictor</h1>
    <p>Enter an individual customer's details to instantly receive a churn probability score from both models.</p>
</div>
""", unsafe_allow_html=True)

log, dt, minmax = load_saved_models()
if log is None:
    st.warning("No trained models found. Please train models in Model Training first."); st.stop()

data_dir = os.path.join(ROOT, "data")
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")] if os.path.exists(data_dir) else []
if not csv_files:
    st.warning("No dataset found. Please upload a CSV in Dataset Explorer first."); st.stop()

with open(os.path.join(data_dir, csv_files[0]), "rb") as f:
    raw_df, df = load_data(f)

_, _, num_cols, cat_cols = clean_data(df)
_, encoders = encode_features(df.drop('Exited', axis=1).copy(), cat_cols)

st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05,
                          help="Probability above which a customer is flagged as High Risk.")
    st.markdown("---")

c1, c2, c3 = st.columns(3)
with c1:
    credit_score    = st.slider("Credit Score",          300, 850, 650)
    age             = st.slider("Age",                    18,  100,  35)
    tenure          = st.slider("Tenure (years)",          0,   10,   5)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])

with c2:
    geography   = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender      = st.selectbox("Gender",    ["Female", "Male"])
    has_cr_card = st.selectbox("Has Credit Card",  [1, 0], format_func=lambda x: "Yes" if x else "No")
    is_active   = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")

with c3:
    balance          = st.number_input("Balance ($)",          0.0, value=0.0,     step=1000.0)
    estimated_salary = st.number_input("Estimated Salary ($)", 0.0, value=50000.0, step=1000.0)

predict_btn = st.button("Predict Churn Risk", use_container_width=True, type="primary")

if predict_btn:
    input_data = pd.DataFrame([{
        'CreditScore': credit_score, 'Geography': geography, 'Gender': gender,
        'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': num_of_products, 'HasCrCard': has_cr_card,
        'IsActiveMember': is_active, 'EstimatedSalary': estimated_salary,
    }])

    for col in cat_cols:
        if col in input_data.columns and col in encoders:
            try:    input_data[col] = encoders[col].transform(input_data[col])
            except: input_data[col] = 0

    for col in num_cols:
        if col in input_data.columns:
            input_data[col] = np.log1p(input_data[col])

    ordered_cols = list(minmax.feature_names_in_)
    input_data   = input_data[ordered_cols]
    X_live = pd.DataFrame(minmax.transform(input_data), columns=ordered_cols)

    lr_prob = log.predict_proba(X_live)[0][1]
    dt_prob = dt.predict_proba(X_live)[0][1]

    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for col, prob, label in [(col1, lr_prob, "Logistic Regression"),
                              (col2, dt_prob, "Decision Tree")]:
        is_high   = prob >= threshold
        card_cls  = "result-high" if is_high else "result-low"
        risk_lbl  = "HIGH RISK — Likely to Churn" if is_high else "LOW RISK — Likely to Stay"
        bar_color = "#f43f5e" if is_high else "#22c55e"

        with col:
            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div class="result-pct">{prob*100:.1f}%</div>
                <div class="result-label"><b>{label}</b><br>{risk_lbl}</div>
            </div>
            <div class="gauge-bg">
                <div class="gauge-fill" style="width:{prob*100:.1f}%; background:{bar_color};"></div>
            </div>
            <p style="text-align:center; font-size:0.78rem; color:#94a3b8; margin-top:4px;">
                Threshold: {threshold*100:.0f}% &nbsp;|&nbsp; Score: {prob*100:.1f}%
            </p>
            """, unsafe_allow_html=True)

    avg_prob = (lr_prob + dt_prob) / 2
    st.markdown("---")
    if avg_prob >= threshold:
        st.error(f"Ensemble Average: {avg_prob*100:.1f}% — This customer should be flagged for a retention intervention.")
    else:
        st.success(f"Ensemble Average: {avg_prob*100:.1f}% — This customer appears satisfied and is unlikely to churn.")
