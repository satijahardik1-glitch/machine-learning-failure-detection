from __future__ import annotations

import csv
import io

import pandas as pd
import plotly.express as px
import streamlit as st

from app.inference import FEATURES, RESERVED_OUTPUT_COLUMNS, predict_batch

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000


def _validate_csv_header(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise ValueError("The upload is not a readable UTF-8 CSV file.") from exc
    if len(set(header)) != len(header):
        raise ValueError("The CSV contains duplicate column names. Make every header unique.")
    missing = [feature for feature in FEATURES if feature not in header]
    if missing:
        raise ValueError("Required sensor column(s) are missing: " + ", ".join(missing))
    reserved = sorted(RESERVED_OUTPUT_COLUMNS.intersection(header))
    if reserved:
        raise ValueError("Remove prediction output column(s) before upload: " + ", ".join(reserved))


def render(models, metadata):
    st.markdown("<div class='eyebrow'>BATCH INFERENCE · CSV</div>", unsafe_allow_html=True)
    st.title("Batch prediction")
    st.write("Upload a CSV with the five sensor columns. Additional columns are preserved in the result.")
    st.caption("Files are processed locally in this app session. Invalid rows are retained and marked; they are not silently imputed.")
    uploaded = st.file_uploader("Sensor readings CSV", type=["csv"], max_upload_size=10)
    if uploaded is None:
        st.info("Required columns: " + " · ".join(FEATURES))
        return

    payload = uploaded.getvalue()
    if len(payload) > MAX_UPLOAD_BYTES:
        st.error("The file exceeds the 10 MB upload limit.")
        return
    try:
        _validate_csv_header(payload)
        source = pd.read_csv(io.BytesIO(payload), low_memory=False)
        if source.empty:
            st.error("The CSV has a header but no data rows.")
            return
        if len(source) > MAX_ROWS:
            st.error(f"The file has {len(source):,} rows; the limit is {MAX_ROWS:,}.")
            return
        result = predict_batch(source, models, metadata)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"CSV could not be processed: {exc}")
        return
    except Exception as exc:
        st.error(f"Batch prediction failed safely: {exc}")
        return

    invalid_count = int(result["Input validation error"].notna().sum())
    valid_count = len(result) - invalid_count
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows received", f"{len(result):,}")
    col2.metric("Rows assessed", f"{valid_count:,}")
    col3.metric("Rows marked invalid", f"{invalid_count:,}")
    st.dataframe(result, width="stretch", hide_index=True)

    valid_results = result.loc[result["Machine condition"].notna()]
    if not valid_results.empty:
        summary = valid_results["Machine condition"].value_counts().rename_axis("Condition").reset_index(name="Rows")
        fig = px.bar(summary, x="Condition", y="Rows", color="Condition", text="Rows",
                     color_discrete_map={"FAILURE RISK":"#E87862","ANOMALOUS CONDITION":"#E7AE59","NORMAL CONDITION":"#42B8A8"})
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.session_state["latest_prediction_summary"] = {
            "condition": "BATCH SUMMARY",
            "summary": ", ".join(f"{row.Condition}: {row.Rows}" for row in summary.itertuples(index=False)) + f" across {valid_count:,} valid rows.",
        }
    st.download_button(
        "Download predictions CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="machine_fault_predictions.csv",
        mime="text/csv",
        type="primary",
        width="stretch",
    )
