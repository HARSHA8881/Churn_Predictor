import streamlit as st
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.styles import inject_css

st.set_page_config(page_title="About | Churn Predictor", page_icon="A", layout="wide")
inject_css()

st.markdown("""
<div class="page-header">
    <h1>About This Project</h1>
    <p>Understanding the problem, methodology, and technology stack behind the Customer Churn Prediction Framework.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>Problem Statement</h3>
        <p>Customer attrition (churn) is one of the most significant expenses for financial institutions.
        The cost of acquiring a new customer is substantially higher than retaining an existing one.
        This project provides a robust, automated classical machine learning framework that predicts
        the likelihood of a customer leaving the bank based on historical demographic and financial behaviour data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>ML Pipeline — 6 Phases</h3>
        <p>
            <span class="phase-badge">Phase 1</span> <b>Raw Data Ingestion</b> — Pandas CSV reader, drops non-predictive ID columns<br><br>
            <span class="phase-badge">Phase 2</span> <b>Imputation</b> — Mean for numerics, Most-Frequent for categoricals via SimpleImputer<br><br>
            <span class="phase-badge">Phase 3</span> <b>Encoding</b> — LabelEncoder converts Geography and Gender to integers<br><br>
            <span class="phase-badge">Phase 4</span> <b>Scaling</b> — <code>log1p</code> transform then MinMaxScaler squeezes features to [0, 1]<br><br>
            <span class="phase-badge">Phase 5</span> <b>Training</b> — 70/30 split; Logistic Regression and Decision Tree trained simultaneously<br><br>
            <span class="phase-badge">Phase 6</span> <b>Serialisation</b> — joblib persists models to the <code>models/</code> directory
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>Key Features</h3>
        <ul>
            <li><b>Dataset Explorer</b> — Upload CSV and explore data with EDA charts</li>
            <li><b>Model Training</b> — One-click training with live progress bar</li>
            <li><b>Performance</b> — Metrics, confusion matrices, ROC curves, feature importance</li>
            <li><b>Business Impact</b> — Translate ML metrics into revenue figures</li>
            <li><b>Churn Predictor</b> — Live single-customer risk assessment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>Tech Stack</h3>
        <div>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">Scikit-Learn</span>
            <span class="tech-pill">Pandas</span>
            <span class="tech-pill">NumPy</span>
            <span class="tech-pill">Matplotlib</span>
            <span class="tech-pill">Seaborn</span>
            <span class="tech-pill">joblib</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
