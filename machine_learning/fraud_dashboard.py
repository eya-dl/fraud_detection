"""Tableau de bord Fraud Analytics : lit les runs MLflow des deux notebooks.
Installer :  pip install -U streamlit plotly mlflow pandas
Lancer :     streamlit run fraud_dashboard.py   (depuis le dossier qui contient mlflow.db)
"""
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlflow.artifacts import download_artifacts

st.set_page_config(page_title="Fraud Analytics", page_icon="🛡️", layout="wide")

TAGS = ["family", "model", "strategy", "purpose", "contamination", "is_reference",
        "feature_set", "dataset_version", "batch_id"]
BUDGET_METRICS = ["Recall_top_0p1pct", "Recall_top_0p5pct", "Recall_top_1pct"]

with st.sidebar:
    st.title("🛡️ Fraud Analytics")
    uri = st.text_input("MLflow tracking URI", "sqlite:///mlflow.db")
    exp_name = st.text_input("Expérience", "Fraud_Detection_Experiments")
    if st.button("Rafraîchir les données"):
        st.cache_data.clear()


@st.cache_data(ttl=60, show_spinner="Lecture de MLflow...")
def load_runs(uri, exp_name):
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        return pd.DataFrame()
    runs = mlflow.search_runs([exp.experiment_id])
    if runs.empty:
        return runs
    df = pd.DataFrame({"run_id": runs["run_id"]})
    for t in TAGS:
        df[t] = runs.get(f"tags.{t}")
    for c in runs.columns:
        if c.startswith("metrics."):
            df[c[8:]] = runs[c]
    df["contamination"] = pd.to_numeric(df["contamination"], errors="coerce")
    df["label"] = np.where(df["family"] == "supervised",
                           df["model"].astype(str) + " | " + df["strategy"].astype(str),
                           df["model"].astype(str) + " | c=" + df["contamination"].astype(str))
    return df


@st.cache_data(show_spinner=False)
def artifact_csv(uri, run_id, name):
    mlflow.set_tracking_uri(uri)
    try:
        return pd.read_csv(download_artifacts(run_id=run_id, artifact_path=f"evaluation/{name}"))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def artifact_path(uri, run_id, name):
    mlflow.set_tracking_uri(uri)
    try:
        return download_artifacts(run_id=run_id, artifact_path=f"evaluation/{name}")
    except Exception:
        return None


runs = load_runs(uri, exp_name)
if runs.empty:
    st.error("Aucun run trouvé. Vérifiez l'URI MLflow et le nom de l'expérience, "
             "puis exécutez les deux notebooks.")
    st.stop()

# ---------------- Filtres
with st.sidebar:
    st.subheader("Filtres")
    versions = sorted(runs["dataset_version"].dropna().unique())
    if not versions:
        st.error("Aucun run avec le tag dataset_version (exécutez les notebooks corrigés).")
        st.stop()
    version = st.selectbox("Version des données", versions, index=len(versions) - 1)
    families = st.multiselect("Famille", ["supervised", "unsupervised"],
                              default=["supervised", "unsupervised"])
    only_ref = st.checkbox("Runs de référence uniquement", value=True,
                           help="Non supervisé : une seule contamination par modèle.")

base = runs[(runs["dataset_version"] == version) & runs["family"].isin(families)]
models = sorted(base["model"].dropna().unique())
with st.sidebar:
    chosen_models = st.multiselect("Modèles", models, default=models)
    strategies = sorted(base["strategy"].dropna().unique())
    chosen_strats = st.multiselect("Stratégies", strategies, default=strategies)
base = base[base["model"].isin(chosen_models) & base["strategy"].isin(chosen_strats)]
view = base[base["is_reference"] == "true"] if only_ref else base
if view.empty:
    st.warning("Aucun run avec ces filtres.")
    st.stop()

# ---------------- En-tête et KPI
st.title("Détection de fraude : tableau de bord des modèles")
best = view.sort_values("PR_AUC", ascending=False).iloc[0]
best_budget = view.sort_values("Recall_top_0p5pct", ascending=False).iloc[0]
best_amount = view.sort_values("Amount_Recall", ascending=False).iloc[0]
fastest = view.sort_values("Training_time_sec").iloc[0]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Runs affichés", len(view))
k2.metric("Meilleure PR-AUC", f"{best['PR_AUC']:.3f}", best["label"], delta_color="off")
k3.metric("Meilleur rappel (budget 0,5 %)", f"{best_budget['Recall_top_0p5pct']:.1%}",
          best_budget["label"], delta_color="off")
k4.metric("Meilleur montant retrouvé", f"{best_amount['Amount_Recall']:.1%}",
          best_amount["label"], delta_color="off")
k5.metric("Entraînement le plus rapide", f"{fastest['Training_time_sec']:.1f} s",
          fastest["label"], delta_color="off")

tabs = st.tabs(["📊 Vue d'ensemble", "⚖️ Comparaison", "📈 Courbes", "🎚️ Seuil et coût",
                "🔎 Détail d'un run", "🧪 Non supervisé"])

# ---------------- 1. Vue d'ensemble
with tabs[0]:
    cols = ["label", "family", "PR_AUC", "ROC_AUC", "Precision", "Recall", "F1", "MCC",
            "F1_tuned", "Amount_Recall", "Recall_top_0p5pct", "Alert_rate", "Training_time_sec"]
    table = view[cols].sort_values("PR_AUC", ascending=False)
    pb = lambda hi=1.0: st.column_config.ProgressColumn(min_value=0, max_value=hi, format="%.3f")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "PR_AUC": pb(), "ROC_AUC": pb(), "F1": pb(), "F1_tuned": pb(),
        "Amount_Recall": pb(), "Recall_top_0p5pct": pb()})
    c1, c2 = st.columns(2)
    fig = px.scatter(view, x="Training_time_sec", y="PR_AUC", color="family", symbol="strategy",
                     size=view["Recall_top_0p5pct"].fillna(0.01).clip(lower=0.01),
                     hover_name="label", log_x=True,
                     title="Performance vs temps d'entraînement (taille = rappel au budget 0,5 %)")
    c1.plotly_chart(fig, width="stretch")
    sup = view[view["family"] == "supervised"]
    if not sup.empty:
        heat = sup.pivot_table(index="model", columns="strategy", values="PR_AUC")
        c2.plotly_chart(px.imshow(heat, text_auto=".3f", color_continuous_scale="Blues",
                                  title="PR-AUC : modèle x stratégie (supervisé)"),
                        width="stretch")
    st.download_button("Exporter le classement (CSV)", table.to_csv(index=False).encode(),
                       "leaderboard.csv", "text/csv")

# ---------------- 2. Comparaison
with tabs[1]:
    all_metrics = [c for c in view.columns if c not in TAGS + ["run_id", "label"]
                   and view[c].notna().any() and view[c].between(0, 1).all()]
    default = [m for m in ["PR_AUC", "F1", "F1_tuned", "Recall_top_0p5pct", "Amount_Recall"]
               if m in all_metrics]
    metrics = st.multiselect("Métriques (échelle 0 à 1)", all_metrics, default=default)
    picked = st.multiselect("Runs à comparer", view["label"].tolist(),
                            default=view.sort_values("PR_AUC", ascending=False)["label"].head(6).tolist())
    sel = view[view["label"].isin(picked)]
    if metrics and not sel.empty:
        long = sel.melt(id_vars="label", value_vars=metrics, var_name="métrique", value_name="valeur")
        st.plotly_chart(px.bar(long, x="métrique", y="valeur", color="label", barmode="group"),
                        width="stretch")
        radar = go.Figure()
        for _, r in sel.iterrows():
            radar.add_trace(go.Scatterpolar(r=[r[m] for m in metrics] + [r[metrics[0]]],
                                            theta=metrics + [metrics[0]], name=r["label"]))
        st.plotly_chart(radar, width="stretch")

# ---------------- 3. Courbes
with tabs[2]:
    kind = st.radio("Courbe", ["Precision-Recall", "ROC"], horizontal=True)
    picked = st.multiselect("Runs", view["label"].tolist(),
                            default=view.sort_values("PR_AUC", ascending=False)["label"].head(5).tolist(),
                            key="curves")
    fig = go.Figure()
    for _, r in view[view["label"].isin(picked)].iterrows():
        d = artifact_csv(uri, r["run_id"], "pr_curve_points.csv" if kind == "Precision-Recall"
                         else "roc_curve_points.csv")
        if d is not None:
            x, y = (d["recall"], d["precision"]) if kind == "Precision-Recall" else (d["fpr"], d["tpr"])
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=r["label"]))
    if kind == "ROC":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="hasard",
                                 line=dict(dash="dash", color="grey")))
    fig.update_layout(xaxis_title="Recall" if kind != "ROC" else "FPR",
                      yaxis_title="Precision" if kind != "ROC" else "TPR", height=550)
    st.plotly_chart(fig, width="stretch")

# ---------------- 4. Seuil et coût
with tabs[3]:
    label = st.selectbox("Run", view.sort_values("PR_AUC", ascending=False)["label"].tolist())
    r = view[view["label"] == label].iloc[0]
    tt = artifact_csv(uri, r["run_id"], "threshold_table.csv")
    if tt is None:
        st.info("threshold_table.csv introuvable pour ce run.")
    else:
        a, b = st.columns(2)
        c_fp = a.number_input("Coût d'une fausse alerte (revue manuelle)", 0.0, value=5.0)
        c_fn = b.number_input("Coût d'une fraude manquée", 0.0, value=500.0)
        tt = tt.assign(cout=tt["FP"] * c_fp + tt["FN"] * c_fn)
        opt = tt.loc[tt["cout"].idxmin()]
        budget = st.slider("Budget d'alertes (% des transactions signalées)", 0.0,
                           float(max(tt["alert_rate"].quantile(0.9) * 100, 1.0)),
                           float(opt["alert_rate"] * 100), 0.05)
        cur = tt.iloc[(tt["alert_rate"] - budget / 100).abs().argsort().iloc[0]]
        m = st.columns(5)
        m[0].metric("Precision", f"{cur['Precision']:.1%}")
        m[1].metric("Recall", f"{cur['Recall']:.1%}")
        m[2].metric("Fraudes retrouvées", f"{int(cur['TP'])}")
        m[3].metric("Fausses alertes", f"{int(cur['FP']):,}")
        m[4].metric("Coût total", f"{cur['cout']:,.0f}")
        fig = go.Figure()
        for c in ["Precision", "Recall", "F1"]:
            fig.add_trace(go.Scatter(x=tt["alert_rate"] * 100, y=tt[c], name=c))
        fig.add_trace(go.Scatter(x=tt["alert_rate"] * 100, y=tt["cout"] / tt["cout"].max(),
                                 name="coût (normalisé)", line=dict(dash="dot")))
        fig.add_vline(x=budget, line_dash="dash")
        fig.update_layout(xaxis_title="% de transactions signalées", xaxis_type="log", height=480)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Coût minimal avec ces coûts : {opt['cout']:,.0f} pour "
                   f"{opt['alert_rate'] * 100:.3f} % d'alertes (precision {opt['Precision']:.1%}, "
                   f"recall {opt['Recall']:.1%}).")

# ---------------- 5. Détail d'un run
with tabs[4]:
    label = st.selectbox("Run", view["label"].tolist(), key="detail")
    r = view[view["label"] == label].iloc[0]
    c1, c2 = st.columns(2)
    for suffix, col in [("", c1), ("_tuned", c2)]:
        p = artifact_path(uri, r["run_id"], f"confusion_matrix{suffix}.png")
        if p:
            col.image(p, caption="Matrice de confusion" + (" (seuil optimisé)" if suffix else " (seuil principal)"))
    imp = artifact_csv(uri, r["run_id"], "feature_importance.csv")
    if imp is not None:
        st.plotly_chart(px.bar(imp.head(15).iloc[::-1], x="importance", y="feature", orientation="h",
                               title="Importance des variables"), width="stretch")
    st.json({k: (None if pd.isna(v) else v) for k, v in r.items() if k not in ("label",)})

# ---------------- 6. Non supervisé
with tabs[5]:
    un = base[base["family"] == "unsupervised"].sort_values("contamination")
    if un.empty:
        st.info("Aucun run non supervisé avec ces filtres.")
    else:
        st.caption("PR-AUC et ROC-AUC ne dépendent pas de la contamination : seul le seuil d'alerte change.")
        long = un.melt(id_vars=["model", "contamination"], value_vars=["Precision", "Recall", "F1"],
                       var_name="métrique", value_name="valeur")
        st.plotly_chart(px.line(long, x="contamination", y="valeur", color="métrique",
                                facet_col="model", markers=True), width="stretch")
        st.dataframe(un[["label", "PR_AUC", "ROC_AUC", "Precision", "Recall", "F1", "Alert_rate",
                         "Amount_Recall"]], hide_index=True, width="stretch")