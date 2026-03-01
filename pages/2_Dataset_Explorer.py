import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data_utils import load_data
from src.styles import inject_css

st.set_page_config(page_title="Dataset Explorer | Churn Predictor", page_icon="D", layout="wide")
inject_css()

st.markdown("""
<div class="page-header">
    <h1>Dataset Explorer</h1>
    <p>Upload your bank customer CSV to preview the data and explore key distributions before training.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Bank Customer CSV (e.g. Churn_Modelling.csv)",
    type=["csv"],
    help="Dataset must contain an 'Exited' column as the target variable."
)

data_dir = os.path.join(ROOT, "data")
os.makedirs(data_dir, exist_ok=True)

if uploaded_file is None:
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if csv_files:
        st.info(f"Auto-loading **{csv_files[0]}** from the data/ folder.")
        with open(os.path.join(data_dir, csv_files[0]), "rb") as f:
            raw_df, df = load_data(f)
    else:
        st.info("Please upload a CSV file above, or place it in the data/ folder.")
        st.stop()
else:
    raw_df, df = load_data(uploaded_file)
    raw_df.to_csv(os.path.join(data_dir, uploaded_file.name), index=False)
    st.success(f"Saved **{uploaded_file.name}** to data/ for use across all pages.")

if 'Exited' not in df.columns:
    st.error("Dataset must contain an 'Exited' column as the target variable.")
    st.stop()

st.markdown('<div class="section-title">Dataset Summary</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
churn_rate = raw_df['Exited'].mean() * 100
missing    = raw_df.isnull().sum().sum()

for col, num, lbl in [
    (c1, f"{len(raw_df):,}",   "Total Customers"),
    (c2, str(raw_df.shape[1]), "Features"),
    (c3, f"{churn_rate:.1f}%", "Churn Rate"),
    (c4, str(missing),         "Missing Values"),
]:
    with col:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
st.dataframe(raw_df.head(15), use_container_width=True)

with st.expander("Descriptive Statistics"):
    st.dataframe(raw_df.describe(), use_container_width=True)

st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

FIG_BG = '#ffffff'

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Overall Churn Distribution**")
    fig1, ax1 = plt.subplots(figsize=(4, 4), facecolor=FIG_BG)
    churn_counts = raw_df['Exited'].value_counts()
    ax1.pie(churn_counts, labels=["Retained", "Churned"], autopct='%1.1f%%',
            colors=['#4361ee', '#f72585'], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2.5})
    ax1.axis('equal')
    st.pyplot(fig1)

with col2:
    st.markdown("**Churn Rate by Geography**")
    fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor=FIG_BG)
    ax2.set_facecolor(FIG_BG)
    sns.countplot(data=raw_df, x='Geography', hue='Exited',
                  palette=['#4361ee', '#f72585'], ax=ax2)
    ax2.set_ylabel("Number of Customers", fontsize=9)
    ax2.legend(["Retained", "Churned"], fontsize=9)
    ax2.tick_params(labelsize=9)
    for spine in ax2.spines.values(): spine.set_color('#e2e8f4')
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Age Distribution by Churn**")
    fig3, ax3 = plt.subplots(figsize=(6, 4), facecolor=FIG_BG)
    ax3.set_facecolor(FIG_BG)
    for v, lbl, clr in [(0, 'Retained', '#4361ee'), (1, 'Churned', '#f72585')]:
        ax3.hist(raw_df[raw_df['Exited'] == v]['Age'], bins=30, alpha=0.7, label=lbl, color=clr)
    ax3.set_xlabel("Age", fontsize=9); ax3.set_ylabel("Count", fontsize=9)
    ax3.legend(fontsize=9)
    for spine in ax3.spines.values(): spine.set_color('#e2e8f4')
    st.pyplot(fig3)

with col4:
    st.markdown("**Balance Distribution by Churn**")
    fig4, ax4 = plt.subplots(figsize=(6, 4), facecolor=FIG_BG)
    ax4.set_facecolor(FIG_BG)
    for v, lbl, clr in [(0, 'Retained', '#4361ee'), (1, 'Churned', '#f72585')]:
        ax4.hist(raw_df[raw_df['Exited'] == v]['Balance'], bins=30, alpha=0.7, label=lbl, color=clr)
    ax4.set_xlabel("Balance ($)", fontsize=9); ax4.set_ylabel("Count", fontsize=9)
    ax4.legend(fontsize=9)
    for spine in ax4.spines.values(): spine.set_color('#e2e8f4')
    st.pyplot(fig4)

st.markdown("**Correlation Heatmap**")
fig5, ax5 = plt.subplots(figsize=(10, 5), facecolor=FIG_BG)
ax5.set_facecolor(FIG_BG)
numeric_df = raw_df.select_dtypes(include=['int64', 'float64'])
for drop_col in ['RowNumber', 'CustomerId']:
    if drop_col in numeric_df.columns:
        numeric_df = numeric_df.drop(drop_col, axis=1)
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5, ax=ax5, annot_kws={"size": 8})
plt.tight_layout()
st.pyplot(fig5)
