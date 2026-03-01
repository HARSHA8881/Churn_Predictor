import streamlit as st
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data_utils import load_data, clean_data, encode_features, scale_features
from src.model import train_models, save_models, evaluate_model
from src.styles import inject_css

st.set_page_config(page_title="Model Training | Churn Predictor", page_icon="M", layout="wide")
inject_css()

st.markdown("""
<div class="page-header">
    <h1>Model Training</h1>
    <p>Run the full ML pipeline to train 4 classifiers with tuned hyperparameters and balanced class weights.</p>
</div>
""", unsafe_allow_html=True)

data_dir = os.path.join(ROOT, "data")
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")] if os.path.exists(data_dir) else []

if not csv_files:
    st.warning("No dataset found. Please upload a CSV in Dataset Explorer first.")
    st.stop()

dataset_path = os.path.join(data_dir, csv_files[0])
st.info(f"Using dataset: **{csv_files[0]}**")

with st.sidebar:
    st.markdown("### Training Config")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")

# ── Pipeline overview ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Pipeline Overview</div>', unsafe_allow_html=True)

phases = [
    ("Phase 1", "Load Data",        "Read CSV, drop RowNumber / CustomerId / Surname"),
    ("Phase 2", "Imputation",       "Mean for numerics · Most-Frequent for categoricals"),
    ("Phase 3", "Encoding",         "LabelEncoder on Geography and Gender"),
    ("Phase 4", "Scaling",          "log1p transform + MinMaxScaler to [0, 1]"),
    ("Phase 5", "Training (x4)",    "Stratified 70/30 split · 4 classifiers with tuned hyperparams"),
    ("Phase 6", "Serialisation",    "joblib dumps 4 .pkl artifacts to models/"),
]

c1, c2 = st.columns(2)
for i, (badge, title, desc) in enumerate(phases):
    col = c1 if i % 2 == 0 else c2
    with col:
        st.markdown(f'<div class="pipeline-step"><b>[{badge}] {title}</b> — {desc}</div>', unsafe_allow_html=True)


# ── Train ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Train Models</div>', unsafe_allow_html=True)

if st.button("Start Training", width="stretch", type="primary"):
    progress = st.progress(0, text="Loading data...")
    with open(dataset_path, "rb") as f:
        raw_df, df = load_data(f)
    progress.progress(10, "Cleaning data...")

    if 'Exited' not in df.columns:
        st.error("Dataset must contain an 'Exited' column."); st.stop()

    X, Y, num_cols, cat_cols = clean_data(df)
    progress.progress(25, "Encoding features...")
    X, encoders = encode_features(X, cat_cols)
    progress.progress(40, "Scaling features...")
    X, minmax = scale_features(X, num_cols)
    progress.progress(55, "Training Logistic Regression + Decision Tree...")

    log, dt, rf, gb, x_train, x_test, y_train, y_test = train_models(X, Y)
    progress.progress(85, "Saving artifacts...")
    save_models(log, dt, rf, gb, minmax)
    progress.progress(100, "Complete!")

    st.session_state['trained']     = True
    st.session_state['log_metrics'] = evaluate_model(log, x_train, y_train, x_test, y_test, threshold)
    st.session_state['dt_metrics']  = evaluate_model(dt,  x_train, y_train, x_test, y_test, threshold)
    st.session_state['rf_metrics']  = evaluate_model(rf,  x_train, y_train, x_test, y_test, threshold)
    st.session_state['gb_metrics']  = evaluate_model(gb,  x_train, y_train, x_test, y_test, threshold)
    st.session_state['threshold']   = threshold
    st.session_state['total_users'] = len(raw_df)

    st.success("All 4 models trained and saved to models/ successfully!")

    st.markdown('<div class="section-title">Quick Results</div>', unsafe_allow_html=True)

    model_results = [
        ("Logistic Regression",  st.session_state['log_metrics']),
        ("Decision Tree",        st.session_state['dt_metrics']),
        ("Random Forest",        st.session_state['rf_metrics']),
        ("Gradient Boosting",    st.session_state['gb_metrics']),
    ]

    col1, col2 = st.columns(2)
    for i, (name, res) in enumerate(model_results):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"**{name}**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{res['Accuracy']:.3f}")
            m2.metric("Precision", f"{res['Precision']:.3f}")
            m3.metric("Recall",    f"{res['Recall']:.3f}")
            m4.metric("F1 Score",  f"{res['F1 Score']:.3f}")

    st.markdown("Navigate to **Performance** in the sidebar for full comparison and charts.")

elif st.session_state.get('trained'):
    st.success("Models are already trained. Navigate to Performance to review results, or click above to retrain.")
else:
    st.info("Click **Start Training** to run the full pipeline on your uploaded dataset.")
