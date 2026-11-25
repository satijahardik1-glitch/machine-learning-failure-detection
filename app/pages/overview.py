from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.inference import TARGET


def render(data, metrics, metadata):
    st.markdown("<div class='eyebrow'>OPERATIONS OVERVIEW · OFFLINE EVALUATION</div>", unsafe_allow_html=True)
    st.title("Machine Fault Intelligence")
    st.markdown(
        "An industrial condition-assessment prototype combining supervised failure risk "
        "with an independent anomaly signal."
    )
    st.info("Historical AI4I dataset and saved holdout evaluation are shown here. This is not a live machine feed.", icon="ℹ️")

    failures = int(data[TARGET].sum())
    total = int(len(data))
    holdout = metrics["evaluation_split"]["holdout_rows"]
    prevalence = failures / total * 100 if total else 0.0
    kpis = st.columns(4)
    kpis[0].metric("Historical observations", f"{total:,}")
    kpis[1].metric("Recorded failures", f"{failures:,}")
    kpis[2].metric("Historical failure share", f"{prevalence:.2f}%")
    kpis[3].metric("Saved holdout size", f"{holdout:,}")

    st.divider()
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Historical failure distribution")
        counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0).rename(index={0: "No recorded failure", 1: "Recorded failure"})
        chart = px.pie(
            values=counts.values,
            names=counts.index,
            hole=0.66,
            color=counts.index,
            color_discrete_map={"No recorded failure": "#3D8291", "Recorded failure": "#E87862"},
        )
        chart.update_layout(height=330, margin=dict(l=8, r=8, t=12, b=8), legend_title_text="")
        st.plotly_chart(chart, width="stretch")
    with right:
        st.subheader("Holdout condition assessment")
        condition_counts = metrics["condition_distribution"]
        condition_frame = {"Condition": list(condition_counts), "Observations": list(condition_counts.values())}
        chart = px.bar(
            condition_frame, x="Observations", y="Condition", orientation="h",
            color="Condition", text="Observations",
            color_discrete_map={"FAILURE RISK": "#E87862", "ANOMALOUS CONDITION": "#E7AE59", "NORMAL CONDITION": "#42B8A8"},
        )
        chart.update_layout(height=330, showlegend=False, margin=dict(l=8, r=8, t=12, b=8))
        st.plotly_chart(chart, width="stretch")
        st.caption("These counts come from the saved 2,500-row evaluation holdout, not live equipment.")

    st.divider()
    st.subheader("Primary model · HistGradientBoosting + SMOTE")
    primary = metrics["models"]["HistGradientBoosting (primary)"]
    cols = st.columns(6)
    for col, label, key, fmt in zip(
        cols,
        ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"],
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        ["{:.1%}", "{:.1%}", "{:.1%}", "{:.3f}", "{:.3f}", "{:.3f}"],
    ):
        col.metric(label, fmt.format(primary[key]))
    st.caption("Saved Phase 2/3 holdout metrics. Model probabilities are scores and have not been separately calibrated.")

    st.subheader("Latest prediction in this session")
    latest = st.session_state.get("latest_prediction_summary")
    if latest:
        st.markdown(f"<div class='status-card'><span class='status-pill'>{latest['condition']}</span><br><br>{latest['summary']}</div>", unsafe_allow_html=True)
    else:
        st.caption("No prediction has been run in this session yet.")

    st.divider()
    st.subheader("Assessment logic")
    status_cols = st.columns(3)
    status_cols[0].markdown("**FAILURE RISK**  \nThe primary supervised model predicts failure.")
    status_cols[1].markdown("**ANOMALOUS CONDITION**  \nNo failure prediction, but the sensor pattern crosses the calibrated anomaly cutoff.")
    status_cols[2].markdown("**NORMAL CONDITION**  \nNeither system raises an alert.")
    st.caption("An anomaly is an unusual pattern, not a confirmed machine fault. This project does not predict remaining useful life.")
