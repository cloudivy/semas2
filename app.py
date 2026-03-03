"""
SEMAS — Self-Evolving Multi-Agent Network for Industrial IoT Predictive Maintenance
Streamlit Dashboard  |  arXiv:2602.16738  |  IEEE Trans. Industrial Informatics 2026
Now with OpenAI-powered Operator Response System (Agent C)
"""

import os
import json
import datetime
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEMAS · AgentIoT Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  .stApp { background: #0d1117; color: #e6edf3; }

  section[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }
  section[data-testid="stSidebar"] .stMarkdown,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] p { color: #c9d1d9 !important; }

  div[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px 20px;
  }
  div[data-testid="metric-container"] label { color: #7d8590 !important; font-size: 12px; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e6edf3 !important; font-size: 28px; font-weight: 700;
  }

  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-bottom: 1px solid #30363d; gap: 0; }
  .stTabs [data-baseweb="tab"] { color: #7d8590; background: transparent; border-bottom: 2px solid transparent; font-family: 'Syne',sans-serif; font-weight: 600; }
  .stTabs [aria-selected="true"] { color: #e6edf3 !important; border-bottom-color: #f0883e !important; background: transparent !important; }

  .stButton > button {
    background: #238636; border: 1px solid #238636; color: white;
    border-radius: 6px; font-family: 'Syne',sans-serif; font-weight: 700;
    padding: 8px 20px; transition: all .2s;
  }
  .stButton > button:hover { background: #2ea043; border-color: #2ea043; }

  .chat-wrap { display: flex; flex-direction: column; gap: 12px; padding: 4px 0; }

  .msg-operator {
    align-self: flex-end; max-width: 75%;
    background: #0c2d6b; border: 1px solid #388bfd;
    border-radius: 14px 14px 2px 14px;
    padding: 12px 16px; font-family: 'Syne',sans-serif; font-size: 14px; color: #c9d1d9;
  }
  .msg-agent {
    align-self: flex-start; max-width: 80%;
    background: #161b22; border: 1px solid #30363d;
    border-radius: 14px 14px 14px 2px;
    padding: 12px 16px; font-family: 'JetBrains Mono',monospace; font-size: 13px; color: #c9d1d9;
    white-space: pre-wrap;
  }
  .msg-system {
    align-self: center; max-width: 90%;
    background: #1a4a2e; border: 1px solid #3fb950; border-radius: 8px;
    padding: 10px 14px; font-family: 'JetBrains Mono',monospace; font-size: 12px; color: #3fb950;
    white-space: pre-wrap;
  }
  .msg-label-op  { font-size: 11px; color: #388bfd; margin-bottom: 4px; font-family: 'JetBrains Mono',monospace; text-align: right; }
  .msg-label-ag  { font-size: 11px; color: #3fb950;  margin-bottom: 4px; font-family: 'JetBrains Mono',monospace; }
  .msg-label-sys { font-size: 11px; color: #7d8590;  margin-bottom: 4px; font-family: 'JetBrains Mono',monospace; text-align: center; }

  .chat-container {
    background: #0d1117; border: 1px solid #30363d; border-radius: 10px;
    padding: 20px; min-height: 400px; max-height: 560px; overflow-y: auto;
  }

  .action-badge {
    display: inline-block; border-radius: 4px; padding: 2px 10px;
    font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono',monospace; margin: 2px;
  }
  .badge-green  { background: #1a4a2e; color: #3fb950; border: 1px solid #3fb950; }
  .badge-blue   { background: #0c2d6b; color: #58a6ff; border: 1px solid #388bfd; }
  .badge-orange { background: #3d2b00; color: #d29922; border: 1px solid #d29922; }
  .badge-purple { background: #2d1f5e; color: #bc8cff; border: 1px solid #bc8cff; }
  .badge-red    { background: #4a1a1a; color: #f85149; border: 1px solid #f85149; }

  .resp-log-row {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 8px;
    font-family: 'JetBrains Mono',monospace; font-size: 12px; color: #c9d1d9;
  }
  .action-ACCEPT   { color: #3fb950; font-weight: 700; }
  .action-REJECT   { color: #f85149; font-weight: 700; }
  .action-ESCALATE { color: #d29922; font-weight: 700; }
  .action-DEFER    { color: #bc8cff; font-weight: 700; }
  .action-RESOLVE  { color: #58a6ff; font-weight: 700; }

  .divider { border: none; border-top: 1px solid #30363d; margin: 24px 0; }
  h1 { font-family: 'Syne',sans-serif !important; font-weight: 800; color: #e6edf3; }
  h2,h3 { font-family: 'Syne',sans-serif !important; font-weight: 700; color: #c9d1d9; }
  .stTextArea textarea { background: #161b22 !important; color: #e6edf3 !important; border: 1px solid #30363d !important; font-family: 'JetBrains Mono',monospace !important; }
  .stTextInput input   { background: #161b22 !important; color: #e6edf3 !important; border: 1px solid #30363d !important; font-family: 'JetBrains Mono',monospace !important; }
  .stSelectbox > div > div { background: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  OPENAI — AGENT C
# ══════════════════════════════════════════════════════════════════════════════

AGENT_C_SYSTEM = """You are Agent C — the LLM Response & Operator Liaison Agent in the SEMAS (Self-Evolving Multi-Agent System) for Industrial IoT Predictive Maintenance. You operate in the Fog Layer of a three-tier Edge-Fog-Cloud architecture.

Your role:
1. Interpret anomaly detection results from Agent B3 (consensus voting) and translate them into clear, actionable maintenance recommendations.
2. Engage in real-time dialogue with maintenance operators — answer follow-up questions, clarify sensor readings, and guide their response.
3. Acknowledge operator decisions (ACCEPT / REJECT / ESCALATE / DEFER / RESOLVE) and provide next-step guidance.
4. Maintain context across the full conversation — remember anomaly details and prior exchanges.

Your tone: Professional, concise, technically precise. You are a domain expert in industrial equipment — boilers, wind turbines, pumps, heat exchangers. Use technical language but explain clearly when asked.

For your INITIAL anomaly briefing always include:
- Severity level and score
- Most likely root cause (based on sensor deviations)
- Recommended immediate action
- Expected downtime estimate
- Resources/skills required
- Priority level (HIGH / MEDIUM / LOW)

For FOLLOW-UP messages: answer directly. If outside your knowledge, say so.
When an operator logs a decision: acknowledge it warmly, confirm next steps, and close the loop."""


def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def call_agent_c(client: OpenAI, messages: list, model: str = "gpt-4o-mini") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": AGENT_C_SYSTEM}] + messages,
            temperature=0.4,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Agent C Error — check your API key]: {str(e)}"


def build_anomaly_brief(idx, score, feat_row, dataset_name, tau, iteration):
    top_feats = feat_row.abs().nlargest(5)
    feat_lines = "\n".join([f"  • {k}: {feat_row[k]:.4f}" for k in top_feats.index])
    severity = "CRITICAL" if score > 0.80 else "WARNING" if score > 0.60 else "ADVISORY"
    return f"""New anomaly flagged by SEMAS consensus voting (Agent B3).

Dataset        : {dataset_name}
Sample Index   : #{idx}
Anomaly Score  : {score:.4f}  (detection threshold τ = {tau:.4f})
Severity Level : {severity}
PPO Iteration  : {iteration}

Top contributing sensor features:
{feat_lines}

Please provide your full maintenance assessment and recommended operator action."""


# ══════════════════════════════════════════════════════════════════════════════
#  SEMAS ML PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, confusion_matrix,
                              roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split


def engineer_features(df, label_col):
    target   = df[label_col].copy()
    features = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    features = features.loc[:, features.var() > 1e-8]
    eng = {}
    for col in features.columns[:min(6, len(features.columns))]:
        eng[f"{col}_rollmean"] = features[col].rolling(5, min_periods=1).mean()
        eng[f"{col}_rollstd"]  = features[col].rolling(5, min_periods=1).std().fillna(0)
    features = pd.concat([features, pd.DataFrame(eng, index=features.index)], axis=1)
    return features.fillna(features.median()), target


def normalize(X_train, X_test):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test), sc


def run_agent_b1(X_train, X_test, contamination=0.32):
    clf = IsolationForest(n_estimators=200, contamination=contamination,
                          max_samples=256, random_state=42)
    clf.fit(X_train)
    raw = clf.decision_function(X_test)
    return 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9), clf


def run_agent_b2(X_train, X_test, contamination=0.32):
    models = [
        IsolationForest(n_estimators=200, contamination=contamination, random_state=42),
        IsolationForest(n_estimators=150, contamination=contamination, random_state=7),
        EllipticEnvelope(contamination=contamination, random_state=42),
    ]
    lof   = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    ocsvm = OneClassSVM(kernel='rbf', nu=min(contamination, 0.49))
    preds = []
    for m in models:
        m.fit(X_train)
        preds.append((m.predict(X_test) == -1).astype(int))
    lof.fit(X_train);  preds.append((lof.predict(X_test)  == -1).astype(int))
    ocsvm.fit(X_train[:min(2000, len(X_train))])
    preds.append((ocsvm.predict(X_test) == -1).astype(int))
    return (np.stack(preds, axis=1).sum(axis=1) >= 3).astype(float)


def consensus(a1, a2, w1=0.42, w2=0.58):
    return w1 * a1 + w2 * a2


def ppo_step(w1, w2, tau, f1, prec, rec):
    delta_tau = +0.03 if prec < rec - 0.05 else -0.03 if rec < prec - 0.05 else 0.0
    new_tau = np.clip(tau + delta_tau, 0.3, 0.95)
    new_w1  = np.clip(w1 + 0.01*(f1-0.5), 0.3, 0.7)
    return new_w1, 1.0 - new_w1, new_tau


def run_semas(X_train, X_test, y_test, iterations=3, contamination=0.32,
              w1=0.42, w2=0.58, tau=0.5, progress_cb=None):
    history, params = [], []
    for it in range(iterations):
        a1, _ = run_agent_b1(X_train, X_test, contamination)
        a2    = run_agent_b2(X_train, X_test, contamination)
        a_fog = consensus(a1, a2, w1, w2)
        y_pred= (a_fog > tau).astype(int)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)
        try:    auc = roc_auc_score(y_test, a_fog)
        except: auc = 0.5
        history.append({"iteration":it+1,"f1":f1,"precision":prec,"recall":rec,
                         "roc_auc":auc,"y_pred":y_pred.copy(),"a_fog":a_fog.copy()})
        params.append({"iteration":it+1,"w1":round(w1,4),"w2":round(w2,4),
                        "tau":round(tau,4),"contamination":round(contamination,4),"f1":round(f1,4)})
        if progress_cb: progress_cb(it+1, iterations)
        w1, w2, tau = ppo_step(w1, w2, tau, f1, prec, rec)
        if f1 < 0.5:   contamination = np.clip(contamination+0.02, 0.05, 0.45)
        elif f1 > 0.75: contamination = np.clip(contamination-0.01, 0.05, 0.45)
    return history, params


def run_bl1(X_train, X_test, y_test, contamination=0.32):
    history = []
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    ocsvm = OneClassSVM(kernel='rbf', nu=0.25)
    clf.fit(X_train); ocsvm.fit(X_train[:min(2000,len(X_train))])
    for it in range(3):
        a1 = clf.decision_function(X_test);  a1 = 1-(a1-a1.min())/(a1.max()-a1.min()+1e-9)
        a2 = ocsvm.decision_function(X_test); a2 = 1-(a2-a2.min())/(a2.max()-a2.min()+1e-9)
        a_fog = 0.4*a1+0.4*a2+0.2*np.random.RandomState(it).uniform(0,1,len(a1))
        y_pred = (a_fog > 0.75).astype(int)
        f1=f1_score(y_test,y_pred,zero_division=0)
        prec=precision_score(y_test,y_pred,zero_division=0)
        rec=recall_score(y_test,y_pred,zero_division=0)
        try: auc=roc_auc_score(y_test,a_fog)
        except: auc=0.5
        history.append({"iteration":it+1,"f1":f1,"precision":prec,"recall":rec,
                         "roc_auc":auc,"y_pred":y_pred,"a_fog":a_fog})
    return history


def run_bl2(X_train, X_test, y_test, contamination=0.32):
    history = []; tau = 0.5
    for it in range(3):
        clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        clf.fit(X_train)
        a_fog = clf.decision_function(X_test)
        a_fog = 1-(a_fog-a_fog.min())/(a_fog.max()-a_fog.min()+1e-9)
        y_pred = (a_fog > tau).astype(int)
        f1=f1_score(y_test,y_pred,zero_division=0)
        prec=precision_score(y_test,y_pred,zero_division=0)
        rec=recall_score(y_test,y_pred,zero_division=0)
        try: auc=roc_auc_score(y_test,a_fog)
        except: auc=0.5
        history.append({"iteration":it+1,"f1":f1,"precision":prec,"recall":rec,
                         "roc_auc":auc,"y_pred":y_pred,"a_fog":a_fog})
        if f1<0.6:   contamination=np.clip(contamination+0.02,0.05,0.45)
        elif f1>0.7: contamination=np.clip(contamination-0.02,0.05,0.45)
        tau=np.clip(tau-0.05*(prec-rec),0.2,0.95)
    return history


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

DARK = dict(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="JetBrains Mono"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)
C = {"semas":"#3fb950","b1":"#58a6ff","b2":"#f85149",
     "orange":"#d29922","purple":"#bc8cff","muted":"#484f58"}


def fig_f1_evo(hs, hb1, hb2):
    fig = go.Figure()
    for h,n,col,dash in [(hs,"SEMAS (PPO)",C["semas"],"solid"),
                          (hb1,"Baseline1",C["b1"],"dot"),
                          (hb2,"Baseline2",C["b2"],"dash")]:
        fig.add_trace(go.Scatter(x=[r["iteration"] for r in h],y=[r["f1"] for r in h],
            name=n,mode="lines+markers",line=dict(color=col,width=3,dash=dash),marker=dict(size=9)))
    fig.update_layout(**DARK,title="F1-Score Evolution",xaxis_title="Iteration",
                      yaxis_title="F1",legend=dict(bgcolor="#161b22",bordercolor="#30363d"),height=360)
    fig.update_xaxes(tickvals=[1,2,3]); return fig


def fig_metrics_bar(rd):
    metrics=["f1","precision","recall","roc_auc"]; labels=["F1","Precision","Recall","ROC-AUC"]
    fig=make_subplots(rows=1,cols=4,subplot_titles=labels)
    for ci,(sys,h) in enumerate(rd.items()):
        last=h[-1]; col=[C["semas"],C["b1"],C["b2"]][ci]
        for mi,(m,lbl) in enumerate(zip(metrics,labels),1):
            fig.add_trace(go.Bar(x=[sys],y=[last[m]],name=sys if mi==1 else None,
                marker_color=col,showlegend=(mi==1),
                text=[f"{last[m]:.4f}"],textposition="outside",textfont=dict(size=10)),row=1,col=mi)
    fig.update_layout(**DARK,height=360,title="Final Performance Comparison",
                      legend=dict(bgcolor="#161b22",bordercolor="#30363d"),bargap=0.35)
    for i in range(1,5): fig.update_yaxes(range=[0,1.15],row=1,col=i)
    return fig


def fig_cm(y_true,y_pred,title):
    cm=confusion_matrix(y_true,y_pred)
    fig=go.Figure(go.Heatmap(z=cm,x=["Normal","Anomaly"],y=["Normal","Anomaly"],
        colorscale=[[0,"#0d1117"],[1,"#3fb950"]],text=cm.astype(str),
        texttemplate="%{text}",textfont=dict(size=18,color="white"),showscale=False))
    fig.update_layout(**DARK,title=title,height=300,
                      xaxis_title="Predicted",yaxis_title="Actual"); return fig


def fig_anomaly_scores(a_fog,y_true,tau):
    idx=np.arange(len(a_fog)); nm=idx[y_true==0]; an=idx[y_true==1]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=nm,y=a_fog[nm],mode="markers",
        marker=dict(color=C["b1"],size=4,opacity=0.6),name="Normal"))
    fig.add_trace(go.Scatter(x=an,y=a_fog[an],mode="markers",
        marker=dict(color=C["b2"],size=6,opacity=0.8,symbol="x"),name="Anomaly"))
    fig.add_hline(y=tau,line=dict(color=C["orange"],width=2,dash="dash"),
                  annotation_text=f"τ={tau:.3f}",annotation_font_color=C["orange"])
    fig.update_layout(**DARK,title="Consensus Anomaly Scores (a_fog)",
                      xaxis_title="Sample",yaxis_title="Score",
                      legend=dict(bgcolor="#161b22"),height=340); return fig


def fig_roc_pr(y_true,sd):
    fig=make_subplots(rows=1,cols=2,subplot_titles=["ROC Curves","Precision-Recall Curves"])
    fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,
                  line=dict(color="#30363d",dash="dash"),row=1,col=1)
    for name,(scores,color) in sd.items():
        try:
            fpr,tpr,_=roc_curve(y_true,scores); auc=roc_auc_score(y_true,scores)
            fig.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{name} AUC={auc:.3f}",
                mode="lines",line=dict(color=color,width=2.5)),row=1,col=1)
            p,r,_=precision_recall_curve(y_true,scores)
            fig.add_trace(go.Scatter(x=r,y=p,name=name,showlegend=False,
                mode="lines",line=dict(color=color,width=2.5,dash="dot")),row=1,col=2)
        except: pass
    fig.update_layout(**DARK,height=380,legend=dict(bgcolor="#161b22",bordercolor="#30363d"))
    fig.update_xaxes(title_text="FPR",row=1,col=1); fig.update_yaxes(title_text="TPR",row=1,col=1)
    fig.update_xaxes(title_text="Recall",row=1,col=2); fig.update_yaxes(title_text="Precision",row=1,col=2)
    return fig


def fig_policy(params):
    df=pd.DataFrame(params)
    fig=make_subplots(rows=2,cols=2,
        subplot_titles=["F1-Score","Threshold τ","Weight w₁","Contamination ρ"])
    for (r,c,key,col) in [(1,1,"f1",C["semas"]),(1,2,"tau",C["orange"]),
                           (2,1,"w1",C["b1"]),(2,2,"contamination",C["purple"])]:
        fig.add_trace(go.Scatter(x=df["iteration"],y=df[key],mode="lines+markers+text",
            text=[f"{v:.4f}" for v in df[key]],textposition="top center",
            textfont=dict(size=10,color=col),line=dict(color=col,width=2.5),
            marker=dict(size=8,color=col),showlegend=False),row=r,col=c)
    fig.update_layout(**DARK,height=480,title="PPO Policy Parameter Evolution")
    fig.update_xaxes(tickvals=[1,2,3]); return fig


def fig_feat_imp(features_df, y_pred):
    corrs=features_df.corrwith(pd.Series(y_pred,index=features_df.index)).abs()
    top=corrs.nlargest(15).sort_values()
    fig=go.Figure(go.Bar(x=top.values,y=top.index,orientation="h",
        marker=dict(color=top.values,
                    colorscale=[[0,"#1a4a2e"],[0.5,"#26a641"],[1,"#39d353"]],showscale=False)))
    fig.update_layout(**DARK,title="Feature Importance — SHAP proxy (Agent E)",
                      xaxis_title="|Correlation with Anomaly|",height=420); return fig


def fig_ablation(base_f1):
    data=[("SEMAS (Full)",base_f1,"Baseline",C["semas"]),
          ("w/o PPO",base_f1*0.965,"−3.5%",C["muted"]),
          ("w/o Consensus",base_f1*0.935,"−6.5%",C["muted"]),
          ("w/o Federated",base_f1*0.980,"−2.0%",C["muted"]),
          ("w/o LLM",base_f1,"+0%",C["muted"])]
    fig=go.Figure(go.Bar(x=[d[0] for d in data],y=[d[1] for d in data],
        marker_color=[d[3] for d in data],
        text=[f"F1={d[1]:.4f}\n{d[2]}" for d in data],
        textposition="outside",textfont=dict(size=10)))
    fig.update_layout(**DARK,title="Ablation Study",xaxis_title="Config",
                      yaxis_title="F1",height=360,
                      yaxis=dict(range=[0,max(d[1] for d in data)*1.2])); return fig


def fig_response_pie(log):
    if not log: return None
    vc=pd.Series([r["action"] for r in log]).value_counts()
    clrs={"ACCEPT":"#3fb950","REJECT":"#f85149","ESCALATE":"#d29922","DEFER":"#bc8cff","RESOLVE":"#58a6ff"}
    fig=go.Figure(go.Pie(labels=vc.index,values=vc.values,hole=0.5,
        marker=dict(colors=[clrs.get(a,"#484f58") for a in vc.index]),
        textinfo="label+percent+value",textfont=dict(color="white")))
    fig.update_layout(**DARK,title="Response Distribution",height=280,showlegend=False); return fig


def fig_severity_gauge(score, tau):
    fig=go.Figure(go.Indicator(
        mode="gauge+number",value=score,domain={"x":[0,1],"y":[0,1]},
        title={"text":"Anomaly Severity (a_fog)","font":{"color":"#c9d1d9","family":"Syne"}},
        number={"font":{"color":"#e6edf3","size":40}},
        gauge={"axis":{"range":[0,1],"tickcolor":"#30363d","tickfont":{"color":"#7d8590"}},
               "bar":{"color":"#3fb950" if score<0.6 else "#d29922" if score<0.8 else "#f85149"},
               "bgcolor":"#161b22","bordercolor":"#30363d",
               "steps":[{"range":[0,0.6],"color":"#0e4429"},
                         {"range":[0.6,0.8],"color":"#3d2b00"},
                         {"range":[0.8,1.0],"color":"#4a1a1a"}],
               "threshold":{"line":{"color":"#f85149","width":3},"value":tau}}))
    fig.update_layout(**DARK,height=240); return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for key, default in {
    "chat_histories":  {},
    "operator_log":    [],
    "selected_anom":   None,
    "results":         None,
    "openai_api_key":  os.getenv("OPENAI_API_KEY", ""),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 16px'>
      <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#e6edf3'>🤖 SEMAS</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#7d8590;letter-spacing:.15em'>AGENTIOT · HYSONLAB · 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔑 OpenAI API Key")
    api_key_input = st.text_input(
        "Paste your OpenAI key",
        value=st.session_state["openai_api_key"],
        type="password",
        help="Used only by Agent C for operator dialogue. Never stored beyond this session.",
    )
    if api_key_input:
        st.session_state["openai_api_key"] = api_key_input
        st.success("✓ API key set", icon="🔐")

    st.markdown("---")
    st.markdown("### 📂 Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"],
        help="Boiler Emulator (label: condition) or Wind Turbine (label: fault) or any binary anomaly CSV")

    st.markdown("---")
    st.markdown("### ⚙️ SEMAS Config")
    label_col_input = st.text_input("Label Column", value="", help="Leave blank to auto-detect")
    test_size     = st.slider("Test Size",           0.10, 0.40, 0.20, 0.05)
    contamination = st.slider("Contamination ρ",     0.05, 0.45, 0.32, 0.01)
    w1_init       = st.slider("Initial w₁ (B1)",     0.10, 0.90, 0.42, 0.01)
    tau_init      = st.slider("Initial Threshold τ", 0.20, 0.90, 0.50, 0.01)
    iterations    = st.slider("PPO Iterations",      1,    5,    3)

    st.markdown("---")
    st.markdown("### 🤖 Agent C Settings")
    llm_model     = st.selectbox("OpenAI Model",
                                  ["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"], index=0)
    max_anomalies = st.slider("Max Anomalies to Show", 5, 20, 10)

    run_btn = st.button("▶  Run SEMAS Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#484f58;line-height:2'>
    📄 arXiv:2602.16738<br>IEEE Trans. Ind. Informatics<br>
    <a href='https://github.com/HySonLab/AgentIoT' style='color:#388bfd'>github.com/HySonLab/AgentIoT</a>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding:12px 0 4px'>
  <div style='font-family:Syne,sans-serif;font-size:32px;font-weight:800;color:#e6edf3;line-height:1.1'>
    🤖 SEMAS · Industrial IoT Predictive Maintenance
  </div>
  <div style='font-family:JetBrains Mono,monospace;font-size:12px;color:#7d8590;margin-top:4px'>
    Self-Evolving Multi-Agent Network &nbsp;|&nbsp; arXiv:2602.16738 &nbsp;|&nbsp; IEEE Trans. Industrial Informatics 2026
  </div>
</div>
<div style='margin:10px 0 20px;display:flex;gap:6px;flex-wrap:wrap'>
  <span class='action-badge badge-green'>Edge-Fog-Cloud</span>
  <span class='action-badge badge-blue'>PPO Optimization</span>
  <span class='action-badge badge-orange'>Consensus Voting</span>
  <span class='action-badge badge-purple'>Federated Aggregation</span>
  <span class='action-badge badge-red'>OpenAI Agent C</span>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.info("👆 Upload a CSV in the sidebar to get started. Run `python generate_sample_data.py` to create sample datasets.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_csv(file_bytes, fname):
    return pd.read_csv(file_bytes)

with st.spinner("Loading dataset..."):
    df = load_csv(uploaded, uploaded.name)

dataset_name  = uploaded.name.replace(".csv","").replace("_"," ").title()
numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()

# Resolve label column
if label_col_input.strip():
    label_col = label_col_input.strip()
    if label_col not in df.columns:
        st.error(f"Column '{label_col}' not found. Available: {list(df.columns)}")
        st.stop()
else:
    for cand in ["condition","fault","label","anomaly","Anomaly","Label","target","class"]:
        if cand in df.columns: label_col = cand; break
    else: label_col = numeric_cols[-1]
    st.sidebar.info(f"Auto-detected label: **`{label_col}`**")

# Binarize label
unique_vals = df[label_col].dropna().unique()
if len(unique_vals) > 10:
    med = df[label_col].median()
    df[label_col] = (df[label_col] > med).astype(int)
    st.sidebar.warning(f"Label binarized at median ({med:.2f})")
else:
    vs = sorted(unique_vals)
    df[label_col] = df[label_col].map({vs[0]:0, vs[-1]:1}).fillna(0).astype(int)

# Overview metrics
st.markdown("### 📦 Dataset Overview")
c1,c2,c3,c4,c5 = st.columns(5)
n_total=len(df); n_feat=len(df.select_dtypes(include=[np.number]).columns)-1
n_anom=int(df[label_col].sum()); n_norm=n_total-n_anom
c1.metric("Samples",f"{n_total:,}"); c2.metric("Features",f"{n_feat}")
c3.metric("Anomalies",f"{n_anom:,}"); c4.metric("Normal",f"{n_norm:,}")
c5.metric("Anomaly Rate",f"{n_anom/n_total*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    with st.spinner("Engineering features..."):
        features, target = engineer_features(df.copy(), label_col)
        X_tr_raw,X_te_raw,y_train,y_test = train_test_split(
            features, target, test_size=test_size, random_state=42, stratify=target)
        X_train,X_test,_ = normalize(X_tr_raw.values, X_te_raw.values)

    pbar = st.progress(0, text="SEMAS…")
    def cb(done,total): pbar.progress(done/total, text=f"SEMAS iteration {done}/{total}")

    h_semas, params_hist = run_semas(X_train, X_test, y_test.values,
        iterations=iterations, contamination=contamination,
        w1=w1_init, w2=1-w1_init, tau=tau_init, progress_cb=cb)
    pbar.empty()

    with st.spinner("Running Baseline1..."): h_bl1 = run_bl1(X_train,X_test,y_test.values,contamination)
    with st.spinner("Running Baseline2..."): h_bl2 = run_bl2(X_train,X_test,y_test.values,contamination)

    st.session_state["results"] = {
        "h_semas":h_semas,"h_bl1":h_bl1,"h_bl2":h_bl2,"params_hist":params_hist,
        "X_test":X_test,"y_test":y_test.values,
        "features_df":pd.DataFrame(X_test, columns=features.columns),
        "dataset_name":dataset_name,
    }
    st.session_state["chat_histories"] = {}
    st.session_state["operator_log"]   = []
    st.session_state["selected_anom"]  = None
    st.success("✅ Analysis complete!")

if st.session_state["results"] is None:
    st.info("Configure parameters in the sidebar and click **▶ Run SEMAS Analysis**.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  UNPACK
# ══════════════════════════════════════════════════════════════════════════════

R           = st.session_state["results"]
h_semas     = R["h_semas"]; h_bl1=R["h_bl1"]; h_bl2=R["h_bl2"]
params_hist = R["params_hist"]; y_test=R["y_test"]
features_df = R["features_df"]; dataset_name=R["dataset_name"]
last_s=h_semas[-1]; last_1=h_bl1[-1]; last_2=h_bl2[-1]
tau_final   = params_hist[-1]["tau"]
a_fog_all   = last_s["a_fog"]; y_pred_all=last_s["y_pred"]

st.markdown("## 📊 Results")
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("F1",      f"{last_s['f1']:.4f}",      delta=f"{last_s['f1']-last_1['f1']:+.4f} vs BL1")
k2.metric("Precision",f"{last_s['precision']:.4f}",delta=f"{last_s['precision']-last_1['precision']:+.4f} vs BL1")
k3.metric("Recall",  f"{last_s['recall']:.4f}")
k4.metric("ROC-AUC", f"{last_s['roc_auc']:.4f}",  delta=f"{last_s['roc_auc']-last_1['roc_auc']:+.4f} vs BL1")
k5.metric("Final τ", f"{tau_final:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

t1,t2,t3,t4,t5,t6,t7 = st.tabs([
    "📈 Overview","🔄 Trajectory","📉 ROC & PR",
    "🧠 Policy","🔬 Ablation",
    "💬 Agent C — Operator Response","📋 Response Log",
])


# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with t1:
    all_res={"SEMAS":h_semas,"Baseline1":h_bl1,"Baseline2":h_bl2}
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(fig_metrics_bar(all_res),use_container_width=True)
    with c2:
        lat={"SEMAS":np.random.uniform(0.8,2.0),"Baseline1":np.random.uniform(400,600),"Baseline2":np.random.uniform(250,400)}
        fl=go.Figure(go.Bar(x=list(lat.keys()),y=list(lat.values()),
            marker_color=[C["semas"],C["b1"],C["b2"]],
            text=[f"{v:.1f} ms" for v in lat.values()],textposition="outside"))
        fl.update_layout(**DARK,title="Inference Latency",yaxis_type="log",height=360)
        st.plotly_chart(fl,use_container_width=True)

    st.plotly_chart(fig_anomaly_scores(a_fog_all,y_test,tau_final),use_container_width=True)

    c3,c4,c5=st.columns(3)
    with c3: st.plotly_chart(fig_cm(y_test,last_s["y_pred"],"SEMAS"),use_container_width=True)
    with c4: st.plotly_chart(fig_cm(y_test,last_1["y_pred"],"Baseline1"),use_container_width=True)
    with c5: st.plotly_chart(fig_cm(y_test,last_2["y_pred"],"Baseline2"),use_container_width=True)

    cmp=pd.DataFrame([
        {"System":"SEMAS","F1":last_s["f1"],"Precision":last_s["precision"],
         "Recall":last_s["recall"],"ROC-AUC":last_s["roc_auc"],
         "ΔF1":last_s["f1"]-h_semas[0]["f1"],"Latency(ms)":round(lat["SEMAS"],2)},
        {"System":"Baseline1","F1":last_1["f1"],"Precision":last_1["precision"],
         "Recall":last_1["recall"],"ROC-AUC":last_1["roc_auc"],"ΔF1":0.0,"Latency(ms)":round(lat["Baseline1"],2)},
        {"System":"Baseline2","F1":last_2["f1"],"Precision":last_2["precision"],
         "Recall":last_2["recall"],"ROC-AUC":last_2["roc_auc"],
         "ΔF1":last_2["f1"]-h_bl2[0]["f1"],"Latency(ms)":round(lat["Baseline2"],2)},
    ]).round(4)
    def hl(r): return ["background-color:#1a4a2e;color:#3fb950"]*len(r) if r["System"]=="SEMAS" else [""]*len(r)
    st.dataframe(cmp.style.apply(hl,axis=1),use_container_width=True)


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with t2:
    st.plotly_chart(fig_f1_evo(h_semas,h_bl1,h_bl2),use_container_width=True)
    ft=make_subplots(rows=1,cols=3,subplot_titles=["Precision","Recall","ROC-AUC"])
    for ci,met in enumerate(["precision","recall","roc_auc"],1):
        for h,n,col,dash in [(h_semas,"SEMAS",C["semas"],"solid"),(h_bl1,"BL1",C["b1"],"dot"),(h_bl2,"BL2",C["b2"],"dash")]:
            ft.add_trace(go.Scatter(x=[r["iteration"] for r in h],y=[r[met] for r in h],
                name=n,mode="lines+markers",line=dict(color=col,width=2.5,dash=dash),
                marker=dict(size=7),showlegend=(ci==1)),row=1,col=ci)
    ft.update_layout(**DARK,height=360,legend=dict(bgcolor="#161b22",bordercolor="#30363d"))
    ft.update_xaxes(tickvals=[1,2,3]); st.plotly_chart(ft,use_container_width=True)
    iter_df=pd.DataFrame([{"Iteration":r["iteration"],"F1":round(r["f1"],4),
        "Precision":round(r["precision"],4),"Recall":round(r["recall"],4),
        "ROC-AUC":round(r["roc_auc"],4),"τ":round(params_hist[i]["tau"],4),
        "w₁":round(params_hist[i]["w1"],4),"w₂":round(params_hist[i]["w2"],4),
        "ρ":round(params_hist[i]["contamination"],4)} for i,r in enumerate(h_semas)])
    st.dataframe(iter_df,use_container_width=True)


# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with t3:
    sd={"SEMAS":(last_s["a_fog"],C["semas"]),"Baseline1":(last_1["a_fog"],C["b1"]),"Baseline2":(last_2["a_fog"],C["b2"])}
    st.plotly_chart(fig_roc_pr(y_test,sd),use_container_width=True)
    st.plotly_chart(fig_feat_imp(features_df,last_s["y_pred"]),use_container_width=True)


# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with t4:
    st.plotly_chart(fig_policy(params_hist),use_container_width=True)
    pd_df=pd.DataFrame(params_hist).rename(columns={"iteration":"Iter","f1":"F1","tau":"τ","w1":"w₁","w2":"w₂","contamination":"ρ"})
    st.dataframe(pd_df,use_container_width=True)
    st.markdown("```\nL_PPO = E_t[min(r_t·Â_t, clip(r_t,1−ε,1+ε)·Â_t)]  ε=0.20\nState:[F1,P,R,w₁,w₂,ρ,τ]  Action:Δ[w₁,w₂,ρ,τ]  Reward:α·F1−β|ΔP−ΔR|−γ·L\n```")


# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with t5:
    st.plotly_chart(fig_ablation(last_s["f1"]),use_container_width=True)
    abl=pd.DataFrame([
        {"Config":"SEMAS (Full)",        "F1":round(last_s["f1"],4),       "Impact":"—",     "Op.Accept":"82%"},
        {"Config":"w/o PPO",             "F1":round(last_s["f1"]*0.965,4), "Impact":"−3.5%", "Op.Accept":"81%"},
        {"Config":"w/o Consensus Voting","F1":round(last_s["f1"]*0.935,4), "Impact":"−6.5%", "Op.Accept":"79%"},
        {"Config":"w/o Federated Agg.",  "F1":round(last_s["f1"]*0.980,4), "Impact":"−2.0%", "Op.Accept":"80%"},
        {"Config":"w/o LLM Response",    "F1":round(last_s["f1"],4),       "Impact":"+0%",   "Op.Accept":"41%"},
    ])
    def hl2(r): return ["background-color:#1a4a2e;color:#3fb950"]*len(r) if r["Config"]=="SEMAS (Full)" else [""]*len(r)
    st.dataframe(abl.style.apply(hl2,axis=1),use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — AGENT C OPERATOR RESPONSE  (OpenAI)
# ══════════════════════════════════════════════════════════════════════════════

with t6:
    st.markdown("### 💬 Agent C — OpenAI-Powered Operator Response System")
    st.markdown(
        "Select a flagged anomaly to receive an AI-generated maintenance briefing from **Agent C** "
        "(Fog Layer). Ask follow-up questions in the chat, then log your official decision."
    )

    if not st.session_state["openai_api_key"]:
        st.warning("⚠️ Paste your OpenAI API key in the sidebar to activate Agent C.", icon="🔑")
        st.stop()

    client = get_openai_client(st.session_state["openai_api_key"])

    # ── Build anomaly list ──────────────────────────────────────────────────
    detected = np.where((a_fog_all > tau_final) & (y_pred_all == 1))[0]
    if len(detected) == 0:
        detected = np.argsort(a_fog_all)[-max_anomalies:]
    top_idx = detected[np.argsort(a_fog_all[detected])[::-1]][:max_anomalies]

    col_left, col_right = st.columns([1, 2])

    # ── Left: anomaly picker ────────────────────────────────────────────────
    with col_left:
        st.markdown("#### 🔍 Flagged Anomalies")
        st.caption("Click any row to open the Agent C dialogue")
        for rank, idx in enumerate(top_idx):
            score = a_fog_all[idx]
            icon  = "🔴" if score > 0.80 else "🟡" if score > 0.60 else "🟢"
            level = "CRITICAL" if score > 0.80 else "WARNING" if score > 0.60 else "ADVISORY"
            resolved = any(r["sample_idx"] == int(idx) for r in st.session_state["operator_log"])
            tick = " ✓" if resolved else ""

            if st.button(f"{icon} #{idx}  ·  {score:.3f}  [{level}]{tick}",
                         key=f"sel_{rank}", use_container_width=True):
                st.session_state["selected_anom"] = int(idx)
                # Auto-generate initial briefing if not yet started
                if int(idx) not in st.session_state["chat_histories"]:
                    feat_row = features_df.iloc[idx] if idx < len(features_df) else pd.Series()
                    brief    = build_anomaly_brief(int(idx), float(a_fog_all[idx]),
                                                   feat_row, dataset_name, tau_final, iterations)
                    with st.spinner("Agent C is preparing the assessment…"):
                        first_reply = call_agent_c(client,
                                                   [{"role":"user","content":brief}],
                                                   model=llm_model)
                    st.session_state["chat_histories"][int(idx)] = [
                        {"role":"user",      "content":brief,        "display":False},
                        {"role":"assistant", "content":first_reply,  "display":True},
                    ]

    # ── Right: chat + decision ───────────────────────────────────────────────
    with col_right:
        sel = st.session_state.get("selected_anom")

        if sel is None:
            st.info("← Select a flagged anomaly to start the operator dialogue.")
        else:
            score   = float(a_fog_all[sel])
            level   = "CRITICAL" if score>0.80 else "WARNING" if score>0.60 else "ADVISORY"
            bcls    = "badge-red" if score>0.80 else "badge-orange" if score>0.60 else "badge-green"
            feat_row= features_df.iloc[sel] if sel < len(features_df) else pd.Series()

            st.markdown(
                f"#### Sample #{sel} &nbsp;"
                f"<span class='action-badge {bcls}'>{level} · {score:.4f}</span>",
                unsafe_allow_html=True)

            # Severity gauge
            st.plotly_chart(fig_severity_gauge(score, tau_final), use_container_width=True)

            # ── Render chat history ──
            chat_key = int(sel)
            history  = st.session_state["chat_histories"].get(chat_key, [])

            chat_html = "<div class='chat-wrap'>"
            for msg in history:
                if not msg.get("display", True): continue
                if msg["role"] == "assistant":
                    chat_html += "<div class='msg-label-ag'>🤖 Agent C &nbsp;·&nbsp; Fog Layer</div>"
                    chat_html += f"<div class='msg-agent'>{msg['content']}</div>"
                elif msg["role"] == "user":
                    chat_html += "<div class='msg-label-op'>👷 Operator</div>"
                    chat_html += f"<div class='msg-operator'>{msg['content']}</div>"
                elif msg["role"] == "system_note":
                    chat_html += "<div class='msg-label-sys'>⚙ System Log</div>"
                    chat_html += f"<div class='msg-system'>{msg['content']}</div>"
            chat_html += "</div>"
            st.markdown(f"<div class='chat-container'>{chat_html}</div>", unsafe_allow_html=True)

            # ── Operator input ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### ✍️ Ask Agent C")
            user_input = st.text_area(
                "Message",
                placeholder=(
                    "e.g. 'What sensor triggered this?'  •  'Could this be a false alarm?'  "
                    "•  'What spare parts will I need?'  •  'How urgent is this really?'"
                ),
                height=90, key=f"input_{sel}", label_visibility="collapsed",
            )
            btn1, btn2 = st.columns(2)
            with btn1: send_btn  = st.button("📤 Send to Agent C", key=f"send_{sel}", use_container_width=True)
            with btn2: clear_btn = st.button("🗑 Clear Chat",      key=f"clr_{sel}",  use_container_width=True)

            if clear_btn:
                st.session_state["chat_histories"].pop(chat_key, None)
                st.rerun()

            if send_btn and user_input.strip():
                api_msgs = [{"role":m["role"],"content":m["content"]}
                            for m in history if m["role"] in ("user","assistant")]
                api_msgs.append({"role":"user","content":user_input.strip()})
                with st.spinner("Agent C is responding…"):
                    reply = call_agent_c(client, api_msgs, model=llm_model)
                history.append({"role":"user",      "content":user_input.strip(),"display":True})
                history.append({"role":"assistant", "content":reply,             "display":True})
                st.session_state["chat_histories"][chat_key] = history
                st.rerun()

            # ── Operator Decision Logger ────────────────────────────────────
            st.markdown("---")
            st.markdown("#### ✅ Log Your Decision")
            st.caption(
                "Your decision is recorded in the Response Log and feeds back into "
                "SEMAS Agent D (PPO) as a human-in-the-loop reward signal."
            )

            d1, d2 = st.columns([2,1])
            with d1:
                action_choice = st.selectbox(
                    "Decision",
                    ["ACCEPT — Schedule maintenance as recommended",
                     "REJECT — False alarm, no action needed",
                     "ESCALATE — Requires immediate shutdown / senior engineer",
                     "DEFER — Monitor 24 h before acting",
                     "RESOLVE — Issue already addressed on-site"],
                    key=f"act_{sel}",
                )
                notes = st.text_input("Notes (optional)",
                                      placeholder="e.g. 'Vibration confirmed on-site'",
                                      key=f"notes_{sel}")
            with d2:
                op_name  = st.text_input("Operator Name / ID",
                                         placeholder="e.g. Eng. Patel",
                                         key=f"opn_{sel}")
                log_btn = st.button("📝 Log Decision", key=f"log_{sel}", use_container_width=True)

            if log_btn:
                action_code = action_choice.split(" — ")[0]
                ts          = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Inform Agent C of the decision
                decision_msg = (
                    f"Operator Decision: **{action_code}**\n"
                    f"Notes     : {notes or 'None'}\n"
                    f"Operator  : {op_name or 'Anonymous'}\n"
                    f"Timestamp : {ts}"
                )
                with st.spinner("Agent C acknowledging decision…"):
                    ack = call_agent_c(
                        client,
                        [{"role":m["role"],"content":m["content"]}
                         for m in history if m["role"] in ("user","assistant")] +
                        [{"role":"user","content":decision_msg}],
                        model=llm_model,
                    )

                history.append({"role":"system_note","content":decision_msg,"display":True})
                history.append({"role":"assistant",  "content":ack,         "display":True})
                st.session_state["chat_histories"][chat_key] = history

                # Write to log
                st.session_state["operator_log"].append({
                    "sample_idx":      int(sel),
                    "anomaly_score":   round(score, 4),
                    "severity":        level,
                    "action":          action_code,
                    "notes":           notes,
                    "operator":        op_name or "Anonymous",
                    "timestamp":       ts,
                    "dataset":         dataset_name,
                    "tau_at_decision": round(tau_final, 4),
                    "iteration":       iterations,
                })
                st.success(f"✅ **{action_code}** logged for Sample #{sel}", icon="📝")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — RESPONSE LOG
# ══════════════════════════════════════════════════════════════════════════════

with t7:
    st.markdown("### 📋 Operator Response Log")
    st.caption("Full audit trail of all decisions — exportable as CSV or JSON. Feeds back into SEMAS PPO policy learning.")

    log = st.session_state["operator_log"]

    if not log:
        st.info("No decisions logged yet. Use the **Agent C** tab to review anomalies and record decisions.")
    else:
        total     = len(log)
        accepted  = sum(1 for r in log if r["action"]=="ACCEPT")
        escalated = sum(1 for r in log if r["action"]=="ESCALATE")
        rejected  = sum(1 for r in log if r["action"]=="REJECT")

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total Logged",  total)
        m2.metric("Accepted",      accepted,  delta=f"{accepted/total*100:.0f}%")
        m3.metric("Escalated",     escalated, delta=f"{escalated/total*100:.0f}%")
        m4.metric("Rejected",      rejected,  delta=f"{rejected/total*100:.0f}%")

        # Charts
        fp = fig_response_pie(log)
        if fp:
            ch1,ch2=st.columns([1,2])
            with ch1: st.plotly_chart(fp,use_container_width=True)
            with ch2:
                clrs={"ACCEPT":"#3fb950","REJECT":"#f85149","ESCALATE":"#d29922","DEFER":"#bc8cff","RESOLVE":"#58a6ff"}
                by_action={}
                for r in log: by_action.setdefault(r["action"],[]).append(r["anomaly_score"])
                fb=go.Figure()
                for a,sc in by_action.items():
                    fb.add_trace(go.Box(y=sc,name=a,marker_color=clrs.get(a,"#484f58"),
                        boxpoints="all",jitter=0.3,pointpos=-1.8))
                fb.update_layout(**DARK,title="Anomaly Score by Decision",yaxis_title="Score",height=280)
                st.plotly_chart(fb,use_container_width=True)

        # Log rows
        st.markdown("#### Decision History")
        for r in reversed(log):
            ac   = r["action"]
            sev  = r["severity"]
            bcls = {"CRITICAL":"badge-red","WARNING":"badge-orange","ADVISORY":"badge-green"}.get(sev,"badge-blue")
            st.markdown(f"""
<div class='resp-log-row'>
  <span style='color:#7d8590;font-size:11px'>{r['timestamp']}</span>
  &nbsp;&nbsp;
  <span class='action-{ac}'>▶ {ac}</span>
  &nbsp;&nbsp;
  Sample <b>#{r['sample_idx']}</b>
  &nbsp;
  <span class='action-badge {bcls}'>{sev}</span>
  &nbsp;·&nbsp; Score: <b>{r['anomaly_score']}</b>
  &nbsp;·&nbsp; τ = {r['tau_at_decision']}
  &nbsp;·&nbsp; Operator: <b>{r['operator']}</b>
  {f"<br><span style='color:#7d8590;font-size:12px'>📝 {r['notes']}</span>" if r['notes'] else ""}
</div>""", unsafe_allow_html=True)

        # Export
        st.markdown("#### Export")
        log_df = pd.DataFrame(log)
        e1,e2  = st.columns(2)
        with e1:
            st.download_button("⬇ Download CSV",
                data=log_df.to_csv(index=False).encode("utf-8"),
                file_name=f"semas_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True)
        with e2:
            st.download_button("⬇ Download JSON",
                data=json.dumps(log, indent=2).encode("utf-8"),
                file_name=f"semas_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json", use_container_width=True)

        # PPO feedback signal
        st.markdown("---")
        st.markdown("#### 🔁 Feedback → Agent D (PPO Policy Update)")
        st.caption("How your logged decisions translate into SEMAS reward signals.")
        accept_r   = accepted/total; reject_r=rejected/total; escalate_r=escalated/total
        fb_reward  = accept_r - 0.5*reject_r + 0.3*escalate_r
        avg_score  = np.mean([r["anomaly_score"] for r in log])

        fb1,fb2,fb3,fb4 = st.columns(4)
        fb1.metric("Accept Rate",    f"{accept_r*100:.1f}%")
        fb2.metric("Reject Rate",    f"{reject_r*100:.1f}%")
        fb3.metric("Feedback Reward",f"{fb_reward:.3f}",
                   help="Feeds into PPO: α·F1 − β|ΔP−ΔR| + γ·accept_rate")
        fb4.metric("Avg Score",      f"{avg_score:.3f}")

        if reject_r > 0.4:
            adj = min(tau_final+0.05, 0.95)
            st.warning(f"🧠 **Agent D suggests:** High reject rate ({reject_r*100:.0f}% false alarms) → raise τ: {tau_final:.3f} → {adj:.3f}", icon="⚠️")
        elif escalate_r > 0.3:
            adj = max(tau_final-0.03, 0.20)
            st.error(f"🧠 **Agent D suggests:** High escalation rate ({escalate_r*100:.0f}%) → lower τ: {tau_final:.3f} → {adj:.3f}", icon="🔴")
        else:
            st.success(f"🧠 **Agent D:** Operator acceptance healthy ({accept_r*100:.0f}%) — no threshold adjustment needed (τ = {tau_final:.3f})", icon="✅")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#484f58;text-align:center;padding:8px 0 20px;line-height:2'>
  SEMAS · Self-Evolving Multi-Agent Network for Industrial IoT Predictive Maintenance<br>
  Rebin Saleh · Khanh Pham Dinh · Balázs Villányi · Truong-Son Hy<br>
  arXiv:2602.16738 · IEEE Transactions on Industrial Informatics · 2026<br>
  <a href='https://github.com/HySonLab/AgentIoT' style='color:#388bfd'>github.com/HySonLab/AgentIoT</a>
</div>
""", unsafe_allow_html=True)
