from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(models, metadata):
    st.markdown("<div class='eyebrow'>EVALUATION · MODEL QUALITY</div>", unsafe_allow_html=True)
    st.title("Model analytics")
    st.write("Saved results from the fixed AI4I holdout. Dashboard inputs do not change or recompute these evaluation metrics.")
    st.warning("The anomaly detector is evaluated against recorded failure as a proxy. Combined OR policies have no continuous ranking score, so ROC-AUC and PR-AUC are not defined for them.")

    results = st.session_state["evaluation_results"]["models"]
    rows=[]
    for name, values in results.items():
        rows.append({"Model":name,**{key:values.get(key) for key in ["accuracy","precision","recall","f1","roc_auc","pr_auc","fpr","fnr"]}})
    metrics_frame=pd.DataFrame(rows).set_index("Model")
    st.subheader("Saved holdout metrics")
    st.dataframe(metrics_frame.style.format("{:.3f}",na_rep="—"),width="stretch")

    shown=metrics_frame.reset_index().melt(
        id_vars="Model",value_vars=["precision","recall","f1","pr_auc"],
        var_name="Metric",value_name="Score"
    ).dropna()
    fig=px.bar(shown,x="Model",y="Score",color="Metric",barmode="group",text_auto=".2f",
               color_discrete_sequence=["#45D6C5","#75A9E6","#E7AE59","#E87862"])
    fig.update_layout(height=430,yaxis_range=[0,1],xaxis_tickangle=-20)
    st.plotly_chart(fig,width="stretch")

    rate_frame=metrics_frame.reset_index().melt(id_vars="Model",value_vars=["fpr","fnr"],var_name="Rate",value_name="Value")
    fig=px.bar(rate_frame,x="Model",y="Value",color="Rate",barmode="group",text_auto=".1%",
               color_discrete_map={"fpr":"#E87862","fnr":"#E7AE59"})
    fig.update_layout(height=380,yaxis_tickformat=".0%",xaxis_tickangle=-20)
    st.subheader("False-alarm and missed-failure rates")
    st.plotly_chart(fig,width="stretch")

    st.subheader("Confusion matrix")
    model_name=st.selectbox("Choose saved model result",list(results))
    cm=results[model_name]["confusion_matrix"]
    matrix=[[cm["tn"],cm["fp"]],[cm["fn"],cm["tp"]]]
    fig=go.Figure(data=go.Heatmap(z=matrix,x=["Predicted normal","Predicted failure"],
        y=["Actual normal","Actual failure"],text=matrix,texttemplate="%{text}",
        colorscale=[[0,"#142632"],[1,"#45D6C5"]],showscale=False))
    fig.update_layout(height=340,margin=dict(l=20,r=20,t=15,b=15))
    st.plotly_chart(fig,width="stretch")

    st.subheader("Sensor feature importance")
    importances=st.session_state["evaluation_results"]["feature_importance"]
    left,right=st.columns(2)
    with left:
        rf=pd.DataFrame({"Sensor":list(importances["Random Forest impurity"]),"Importance":list(importances["Random Forest impurity"].values())}).sort_values("Importance")
        fig=px.bar(rf,x="Importance",y="Sensor",orientation="h",color_discrete_sequence=["#45D6C5"],title="Random Forest · impurity importance")
        st.plotly_chart(fig,width="stretch")
    with right:
        hgb=pd.DataFrame({"Sensor":list(importances["HistGradientBoosting permutation AP decrease"]),"Importance":list(importances["HistGradientBoosting permutation AP decrease"].values())}).sort_values("Importance")
        fig=px.bar(hgb,x="Importance",y="Sensor",orientation="h",color_discrete_sequence=["#E7AE59"],title="HistGradientBoosting · permutation importance")
        st.plotly_chart(fig,width="stretch")
    st.caption("Importance is descriptive for these fitted models; it does not prove that a sensor causes failure.")

    st.subheader("Isolation Forest calibration")
    contamination=pd.DataFrame(st.session_state["evaluation_results"]["isolation_contamination_calibration"])
    fig=px.line(contamination,x="contamination",y=["recall","precision","fpr"],markers=True,
                labels={"value":"Calibration metric","contamination":"Contamination","variable":"Metric"})
    fig.update_layout(height=350,yaxis_range=[0,1])
    st.plotly_chart(fig,width="stretch")
    st.caption(
        f"Selected contamination: {metadata['thresholds']['isolation_forest_contamination']:.3f}; "
        f"saved score cutoff: {metadata['thresholds']['isolation_forest_anomaly_score']:.6f}. "
        "Contamination and threshold were selected using training-only calibration data."
    )

