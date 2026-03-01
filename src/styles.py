"""Shared CSS injected into every page for consistent styling."""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

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

.page-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px; padding: 44px 44px; margin-bottom: 32px; color: white;
    box-shadow: 0 20px 60px rgba(15,52,96,0.25); position: relative; overflow: hidden;
}
.page-header::before {
    content: ''; position: absolute; top: -70px; right: -70px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(0,210,150,0.2) 0%, transparent 70%);
    border-radius: 50%;
}
.page-header h1 { font-size: 2rem; font-weight: 800; margin-bottom: 8px; position: relative; }
.page-header p  { color: rgba(255,255,255,0.72); font-size: 0.95rem; line-height: 1.6; position: relative; }

.section-title {
    font-size: 1.15rem; font-weight: 800; color: #1a1a2e;
    margin: 32px 0 18px; display: flex; align-items: center; gap: 10px;
}
.section-title::after {
    content: ''; flex: 1; height: 2px;
    background: linear-gradient(90deg, #e2e8f4, transparent);
}

.card {
    background: #ffffff; border: 1px solid #e2e8f4; border-radius: 16px;
    padding: 24px; margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(15,52,96,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(15,52,96,0.1); }

.info-card {
    background: #ffffff; border: 1px solid #e2e8f4; border-radius: 16px;
    padding: 24px; margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(15,52,96,0.05);
}
.info-card h3 { font-size: 0.95rem; font-weight: 700; color: #1a1a2e; margin-bottom: 12px; }
.info-card p, .info-card li { font-size: 0.88rem; color: #475569; line-height: 1.75; }
.info-card code { background: #f1f5f9; color: #4361ee; padding: 2px 6px; border-radius: 4px; font-size: 0.82rem; }

.phase-badge {
    display: inline-block; background: linear-gradient(90deg, #4361ee, #7b5ea7);
    color: white; font-size: 0.68rem; font-weight: 700;
    padding: 3px 10px; border-radius: 12px; margin-right: 8px;
}
.tech-pill {
    display: inline-block; background: #eef2ff; color: #4361ee;
    font-size: 0.75rem; font-weight: 600;
    padding: 5px 14px; border-radius: 20px; margin: 4px;
}

.stat-card {
    background: #ffffff; border: 1px solid #e2e8f4; border-radius: 14px;
    padding: 20px; text-align: center;
    box-shadow: 0 2px 10px rgba(15,52,96,0.06);
}
.stat-num { font-size: 1.9rem; font-weight: 800; color: #1a1a2e; }
.stat-lbl { font-size: 0.72rem; font-weight: 700; color: #94a3b8;
            text-transform: uppercase; letter-spacing: 0.9px; margin-top: 4px; }

.pipeline-step {
    background: #ffffff; border-left: 4px solid #4361ee; border-radius: 0 12px 12px 0;
    padding: 13px 18px; margin-bottom: 8px; font-size: 0.88rem; color: #334155;
    box-shadow: 0 2px 8px rgba(15,52,96,0.04);
}
.pipeline-step b { color: #1a1a2e; }

.result-card {
    border-radius: 16px; padding: 30px; text-align: center; margin-bottom: 12px;
}
.result-high { background: linear-gradient(135deg, #fff1f2, #ffe4e6); border: 2px solid #f43f5e; }
.result-low  { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 2px solid #22c55e; }
.result-pct  { font-size: 3rem; font-weight: 800; margin-bottom: 6px; }
.result-high .result-pct { color: #be123c; }
.result-low  .result-pct { color: #15803d; }
.result-label { font-size: 0.95rem; font-weight: 600; color: #475569; }

.gauge-bg   { background: #e2e8f0; border-radius: 8px; height: 14px; overflow: hidden; margin: 8px 0; }
.gauge-fill { height: 100%; border-radius: 8px; }

.impact-card {
    background: #ffffff; border: 1px solid #e2e8f4; border-radius: 16px; padding: 26px;
    box-shadow: 0 2px 12px rgba(15,52,96,0.05);
}
.impact-card h4 { font-size: 1rem; font-weight: 800; color: #1a1a2e; margin-bottom: 16px; }

div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
div[data-testid="stMetric"] { background: #fff; border-radius: 12px; padding: 16px;
                               border: 1px solid #e2e8f4; }
                               
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

.stButton>button {
    border-radius: 10px !important; font-weight: 700 !important;
    transition: all 0.2s !important;
}
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #4361ee, #7b5ea7) !important;
    border: none !important; color: white !important;
}
.stButton>button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(67,97,238,0.4) !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0 !important; }
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
