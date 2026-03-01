import streamlit as st
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import load_saved_models

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background: #f0f2f8 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
[data-testid="stSidebarNavItems"] a[aria-current="page"] {
    background: rgba(67, 97, 238, 0.35) !important;
    border-radius: 8px; color: #fff !important; font-weight: 700 !important;
}
[data-testid="stSidebarNavItems"] a:hover {
    background: rgba(255,255,255,0.1) !important; border-radius: 8px;
}

header[data-testid="stHeader"] {
    background: #f0f2f8 !important;
    border-bottom: 1px solid #dde3f0 !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px;
    padding: 52px 44px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(15, 52, 96, 0.3);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(0,210,150,0.2) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 40%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(67,97,238,0.2) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, #00d296, #00b4d8);
    color: #000;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 5px 14px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 18px;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 14px;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.7);
    max-width: 680px;
    line-height: 1.75;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f4;
    border-radius: 16px;
    padding: 26px 22px;
    box-shadow: 0 2px 12px rgba(15,52,96,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(15,52,96,0.12);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1a1a2e;
    line-height: 1;
    margin-bottom: 8px;
}
.metric-delta {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    display: inline-block;
}
.delta-green  { background: #dcfce7; color: #16a34a; }
.delta-blue   { background: #dbeafe; color: #2563eb; }
.delta-purple { background: #ede9fe; color: #7c3aed; }

.section-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #1a1a2e;
    margin: 40px 0 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #e2e8f4, transparent);
    border-radius: 2px;
}

.qs-item {
    background: #ffffff;
    border: 1px solid #e2e8f4;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    font-size: 0.9rem;
    color: #334155;
    line-height: 1.6;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    box-shadow: 0 2px 8px rgba(15,52,96,0.04);
    transition: box-shadow 0.2s;
}
.qs-item:hover { box-shadow: 0 6px 20px rgba(15,52,96,0.09); }
.qs-num {
    background: linear-gradient(135deg, #4361ee, #7b5ea7);
    color: white;
    font-size: 0.75rem;
    font-weight: 800;
    min-width: 26px; height: 26px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.qs-text b { color: #1a1a2e; }

/* Fix for Input Widgets (Selectbox, NumberInput, etc) */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #ffffff !important;
    border: 1px solid #dde3f0 !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] * { color: #1a1a2e !important; }
div[data-baseweb="input"] input { color: #1a1a2e !important; }
div[data-baseweb="popover"] { background-color: #ffffff !important; }
div[data-baseweb="popover"] li { color: #1a1a2e !important; }
div[data-baseweb="popover"] li:hover { background-color: #f0f2f8 !important; }
label[data-testid="stWidgetLabel"] p { color: #475569 !important; font-weight: 600; }

.footer-note {
    text-align: center;
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 48px;
    padding-top: 24px;
    border-top: 1px solid #e2e8f4;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Churn Predictor")
    st.markdown("---")
    st.markdown(
        "Navigate the pages to explore data, train models, "
        "analyse performance, and predict individual churn risk."
    )
    st.markdown("---")
    st.caption("v1.0-RC1  ·  Streamlit + Scikit-Learn")

st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">Machine Learning &middot; Binary Classification</div>
    <div class="hero-title">Welcome to the Customer<br>Churn Prediction Framework!</div>
    <div class="hero-subtitle">
        This system evaluates the probability of a bank customer leaving.
        Built on classical ML principles, we've integrated Logistic Regression &amp;
        Decision Tree models into a clean, interactive multi-page interface.
        Navigate the sidebar to explore data, train models, and make live predictions.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">System Dashboard</div>', unsafe_allow_html=True)

log, dt, rf, gb, minmax = load_saved_models()
models_ready = log is not None

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Models Trained</div>
        <div class="metric-value">{"4" if models_ready else "0"}</div>
        <span class="metric-delta {"delta-green" if models_ready else "delta-blue"}">
            {"Ready to predict" if models_ready else "Train first"}
        </span>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Best F1 Score</div>
        <div class="metric-value" style="font-size:1.5rem; padding-top:4px;">—</div>
        <span class="metric-delta delta-blue">Train to see score</span>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Version</div>
        <div class="metric-value" style="font-size:1.4rem; padding-top:4px;">v1.0-RC1</div>
        <span class="metric-delta delta-purple">Stable release</span>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Framework</div>
        <div class="metric-value" style="font-size:1.4rem; padding-top:4px;">Streamlit</div>
        <span class="metric-delta delta-blue">+ Scikit-Learn</span>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Quick Start Guide</div>', unsafe_allow_html=True)

steps = [
    ("Dataset Explorer",  "Upload your bank customer CSV to preview the data and explore key EDA visualisations — churn distributions, geography breakdowns and correlation heatmaps."),
    ("Model Training",    "Go to <b>Model Training</b> to run the full 6-phase ML pipeline and train both Logistic Regression and Decision Tree models with a single click."),
    ("Performance",       "Switch to <b>Performance</b> to analyse confusion matrices, ROC curves, feature importance, and compare model metrics side-by-side."),
    ("Business Impact",   "Inside <b>Performance</b>, use the <b>Business Impact</b> tab to translate ML metrics into real revenue figures — revenue saved vs. intervention cost."),
    ("Churn Predictor",   "Open <b>Churn Predictor</b>, fill in an individual customer's profile, and instantly see the churn probability from both models plus an ensemble verdict."),
]

for i, (title, desc) in enumerate(steps, 1):
    st.markdown(f"""
    <div class="qs-item">
        <div class="qs-num">{i}</div>
        <div class="qs-text"><b>{title}</b> — {desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    Built with Streamlit &nbsp;&middot;&nbsp; Scikit-Learn &nbsp;&middot;&nbsp; Pandas &nbsp;&middot;&nbsp; Seaborn
    &nbsp;&middot;&nbsp; Customer Churn Prediction Framework v1.0
</div>""", unsafe_allow_html=True)
