from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.inference import DEFAULT_DATA_PATH, load_artifacts, load_dataset

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "models" / "evaluation_results.json"


def main():
    st.set_page_config(page_title="Machine Fault Intelligence",page_icon="⚙️",layout="wide")
    st.markdown("""
    <style>
    .stApp {background: radial-gradient(circle at 80% 0%, #132630 0%, #0A1118 42%);}
    .block-container {padding-top: 1.4rem; padding-bottom: 2.4rem; max-width: 1480px;}
    [data-testid="stSidebar"] {background: #0D171F; border-right: 1px solid #20323E;}
    .eyebrow {color:#45D6C5; letter-spacing:.16em; font-size:.72rem; font-weight:700; margin-bottom:.35rem;}
    .status-card {padding:1.15rem 1.35rem; background:#12212B; border:1px solid #294351; border-radius:14px; margin:.7rem 0 1rem 0;}
    .status-card.risk {border-left:5px solid #E87862;}
    .status-card.anomaly {border-left:5px solid #E7AE59;}
    .status-card.normal {border-left:5px solid #42B8A8;}
    .status-pill {font-weight:800; letter-spacing:.08em; font-size:1.08rem;}
    div[data-testid="stMetric"] {background:#111D27; border:1px solid #223743; padding:14px 16px; border-radius:12px;}
    div[data-testid="stMetricLabel"] p {color:#91A7B5;}
    div[data-testid="stMetricValue"] {color:#E8F0F6;}
    hr {border-color:#243743;}
    </style>
    """,unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("# ⚙️ MFI")
        st.caption("MACHINE FAULT INTELLIGENCE")
        st.markdown("Industrial failure risk & anomaly triage")

    @st.cache_resource(show_spinner="Loading trained model artifacts…")
    def get_models():
        return load_artifacts()

    @st.cache_data(show_spinner=False)
    def get_data(path: str):
        return load_dataset(path)

    try:
        models,metadata=get_models()
        dataset=get_data(str(DEFAULT_DATA_PATH))
        evaluation=json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.session_state["evaluation_results"]=evaluation
    except Exception as exc:
        st.error(f"The dashboard could not initialize: {exc}")
        st.info("From the project root, install `requirements.txt` and run `python scripts/train_artifacts.py` to generate the local models.")
        st.stop()

    from app.pages.overview import render as overview_page
    from app.pages.predict import render as predict_page
    from app.pages.batch import render as batch_page
    from app.pages.analytics import render as analytics_page

    def overview_route():
        overview_page(dataset,evaluation,metadata)

    def prediction_route():
        predict_page(dataset,models,metadata,evaluation)

    def batch_route():
        batch_page(models,metadata)

    def analytics_route():
        analytics_page(models,metadata)

    pages={
        "Monitor":[
            st.Page(overview_route,title="Overview",icon=":material/dashboard:",default=True),
            st.Page(prediction_route,title="Single prediction",icon=":material/sensors:"),
            st.Page(batch_route,title="Batch prediction",icon=":material/upload_file:"),
            st.Page(analytics_route,title="Model analytics",icon=":material/query_stats:"),
        ]
    }
    page=st.navigation(pages,position="sidebar")
    page.run()


if __name__ == "__main__":
    main()

