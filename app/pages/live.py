from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PAGE_PATH = Path(__file__).resolve().parents[2] / "web" / "index.html"


@st.cache_data(show_spinner=False)
def load_page(path: str) -> str:
    html = Path(path).read_text(encoding="utf-8")
    return html.replace('<html lang="en">', '<html lang="en" data-theme="dark">', 1)


def render():
    st.title("Live predictor")
    st.caption("Runs the HistGradientBoosting and Isolation Forest models in your browser. Move a slider to see the state change.")
    if not PAGE_PATH.exists():
        st.error("The live predictor page is missing. Run `python scripts/build_web.py` to generate it.")
        return
    components.html(load_page(str(PAGE_PATH)), height=1250, scrolling=True)
