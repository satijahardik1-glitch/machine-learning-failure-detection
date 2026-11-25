from __future__ import annotations

import pandas as pd
import streamlit as st

from app.inference import FEATURES, anomaly_context, predict_batch


def render(data, models, metadata, evaluation):
    st.markdown("<div class='eyebrow'>SENSOR INPUT · SINGLE MACHINE</div>", unsafe_allow_html=True)
    st.title("Single prediction")
    st.write("Enter one sensor snapshot. Values are evaluated against the trained models and saved operating thresholds.")
    defaults = data[FEATURES].median(numeric_only=True)
    with st.form("single_prediction_form"):
        cols = st.columns(5)
        values = {}
        for col, feature in zip(cols, FEATURES):
            default = float(defaults[feature])
            if feature in ("Rotational speed [rpm]", "Tool wear [min]"):
                values[feature] = col.number_input(feature, min_value=0, value=int(round(default)), step=1, help=f"Unit: {'rpm' if 'rpm' in feature else 'min'}")
            else:
                values[feature] = col.number_input(feature, min_value=0.0, value=default, step=0.1, format="%.2f", help="Unit: Kelvin (K)" if "[K]" in feature else "Unit: Nm")
        submitted = st.form_submit_button("Assess machine condition", type="primary", width="stretch")

    if not submitted:
        st.caption("Defaults use the historical dataset median. Replace them with a real sensor snapshot before interpreting a prediction.")
        return

    source = pd.DataFrame([values], columns=FEATURES)
    try:
        result = predict_batch(source, models, metadata).iloc[0]
    except (TypeError, ValueError) as exc:
        st.error(f"Input could not be evaluated: {exc}")
        return
    if pd.notna(result["Input validation error"]):
        st.error(str(result["Input validation error"]))
        return

    condition = str(result["Machine condition"])
    colors = {"FAILURE RISK": "risk", "ANOMALOUS CONDITION": "anomaly", "NORMAL CONDITION": "normal"}
    st.markdown(f"<div class='status-card {colors.get(condition, '')}'><span class='status-pill'>{condition}</span><p>Assessment from the supervised failure classifier and the separate Isolation Forest score.</p></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("HGB failure model output", f"{float(result['HGB failure probability']):.1%}", help="Model probability output; not separately calibrated.")
    c2.metric("Random Forest comparison", f"{float(result['Random Forest failure probability']):.1%}")
    c3.metric("Isolation Forest anomaly score", f"{float(result['Isolation Forest anomaly score']):.4f}", help=f"Alert cutoff: {metadata['thresholds']['isolation_forest_anomaly_score']:.6f}")
    st.write(f"**Model decisions:** HGB = {'failure risk' if int(result['HGB failure prediction']) else 'no failure prediction'}; Random Forest = {'failure' if int(result['Random Forest failure prediction']) else 'no failure'}; Isolation Forest = {'anomaly' if int(result['Isolation Forest alert']) else 'not flagged'}.")

    st.subheader("Sensor context")
    notes = anomaly_context(source.iloc[0], metadata)
    if int(result["Isolation Forest alert"]):
        st.warning("Isolation Forest identified an unusual multivariate pattern. This does not by itself mean a failure has occurred.")
    else:
        st.caption("The sample did not cross the calibrated Isolation Forest threshold.")
    for note in notes:
        st.markdown(f"- {note}")

    importance = evaluation["feature_importance"]["HistGradientBoosting permutation AP decrease"]
    top_features = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:3]
    st.caption("Dataset-level feature importance is descriptive, not a causal explanation of this individual prediction.")
    st.write("Highest global permutation-importance sensors: " + ", ".join(name for name, _ in top_features) + ".")
    st.session_state["latest_prediction_summary"] = {
        "condition": condition,
        "summary": f"HGB output {float(result['HGB failure probability']):.1%}; anomaly score {float(result['Isolation Forest anomaly score']):.4f}.",
    }
