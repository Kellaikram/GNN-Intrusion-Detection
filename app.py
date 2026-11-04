import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import altair as alt
from sklearn.neighbors import NearestNeighbors

from src.models import GAT, GraphSAGE

K = 5
MAX_ROWS = 5000

st.set_page_config(page_title="GNN IDS", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load("models/scaler.joblib")
    encoders = joblib.load("models/encoders.joblib")
    columns = joblib.load("models/feature_columns.joblib")
    models = {}
    for name, cls in [("GraphSAGE", GraphSAGE), ("GAT", GAT)]:
        model = cls()
        model.load_state_dict(torch.load(f"models/{name}.pt", map_location="cpu"))
        model.eval()
        models[name] = model
    return scaler, encoders, columns, models


def preprocess(df, scaler, encoders, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    x = df[columns].copy()
    for col, le in encoders.items():
        x[col] = x[col].astype(str)
        unknown = set(x[col]) - set(le.classes_)
        if unknown:
            raise ValueError(f"Valeurs inconnues dans '{col}' : {sorted(unknown)}")
        x[col] = le.transform(x[col])
    return scaler.transform(x)


def build_edge_index(x, k=K):
    k = min(k, len(x) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(x)
    _, idx = nbrs.kneighbors(x)
    src = np.repeat(np.arange(len(x)), k)
    dst = idx[:, 1:].reshape(-1)
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)

def read_any_csv(file, columns):
    head = pd.read_csv(file, header=None, nrows=1)
    file.seek(0)
    if str(head.iloc[0, 0]) == columns[0]:
        return pd.read_csv(file)  # fih header

    df = pd.read_csv(file, header=None)  # bla header (KDDTest+.txt)
    n = df.shape[1]
    if n not in (41, 42, 43):
        raise ValueError(f"Fichier avec {n} colonnes : il en faut 41, 42 ou 43.")
    names = columns + ["attack_name", "difficulty"][: n - 41]
    df.columns = names
    if "attack_name" in df.columns:
        is_normal = df["attack_name"].astype(str).str.strip(".") == "normal"
        df["label_reel"] = np.where(is_normal, "Normal", "Attack")
        df = df.drop(columns=["attack_name"])
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])
    return df


def drop_unknown(df, encoders):
    before = len(df)
    for col, le in encoders.items():
        df = df[df[col].astype(str).isin(le.classes_)]
    return df, before - len(df)


def predict(model, x_scaled):
    x = torch.tensor(x_scaled, dtype=torch.float)
    edge_index = build_edge_index(x_scaled)
    with torch.no_grad():
        probs = torch.softmax(model(x, edge_index), dim=1)
    return probs.argmax(dim=1).numpy(), probs[:, 1].numpy()


scaler, encoders, columns, models = load_artifacts()

st.title("🛡️ GNN Intrusion Detection System")
st.caption("Classification du trafic réseau (Normal / Attack) avec GraphSAGE et GAT, entraînés sur KDD99 et tester sur KDDTest+.")

tab_detect, tab_perf = st.tabs(["Détection", "Performance des modèles"])

with tab_detect:
    model_name = st.sidebar.selectbox("Modèle", list(models.keys()))
    source = st.sidebar.radio("Source des données", ["Exemple (2000 lignes)", "Upload CSV"])

    df = None
    if source == "Exemple (2000 lignes)":
        df = pd.read_csv("data/sample.csv")
    else:
        file = st.file_uploader("Fichier KDD99 / NSL-KDD (.csv ou .txt)", type=["csv", "txt"])
        if file:
            try:
                df = read_any_csv(file, columns)
            except ValueError as e:
                st.error(str(e))

    if df is not None:
        df, dropped = drop_unknown(df, encoders)
        if dropped:
            st.warning(f"{dropped} lignes ignorées (valeurs inconnues du modèle).")
        if len(df) > MAX_ROWS:
            st.warning(f"Échantillon aléatoire de {MAX_ROWS} lignes.")
            df = df.sample(MAX_ROWS, random_state=42)
        if len(df) < 2:
            st.error("Il faut au moins 2 lignes.")
        else:
            try:
                x_scaled = preprocess(df, scaler, encoders, columns)
            except ValueError as e:
                st.error(str(e))
            else:
                preds, probs = predict(models[model_name], x_scaled)
                result = df.copy()
                result["prediction"] = np.where(preds == 1, "Attack", "Normal")
                result["proba_attack"] = probs.round(4)

                n_attack = int((preds == 1).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Flux analysés", len(result))
                c2.metric("Attaques détectées", n_attack)
                c3.metric("Taux d'attaque", f"{n_attack / len(result):.1%}")
                if "label_reel" in result.columns:
                    acc = (result["prediction"] == result["label_reel"]).mean()
                    st.metric("Accuracy vs label_reel", f"{acc:.1%}")

                counts = (
                    result["prediction"]
                    .value_counts()
                    .rename_axis("Classe")
                    .reset_index(name="Flux")
                )
                chart = (
                    alt.Chart(counts)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("Classe:N", title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("Flux:Q", title="Nombre de flux"),
                        color=alt.Color(
                            "Classe:N",
                            scale=alt.Scale(
                                domain=["Normal", "Attack"],
                                range=["#22c55e", "#ef4444"],
                            ),
                            legend=None,
                        ),
                        tooltip=["Classe", "Flux"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, width="stretch")

                first = ["prediction", "proba_attack"]
                result = result[first + [c for c in result.columns if c not in first]]

                def color_pred(v):
                    if v == "Attack":
                        return "background-color: #7f1d1d; color: #fecaca"
                    return "background-color: #14532d; color: #bbf7d0"

                st.caption(
                    f"Aperçu des 1000 premières lignes sur {len(result)} "
                    "(le CSV téléchargeable contient tout)."
                )
                st.dataframe(
                    result.head(1000).style.map(color_pred, subset=["prediction"]),
                    width="stretch",
                )
                st.download_button(
                    "Télécharger les résultats (CSV)",
                    result.to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                )

with tab_perf:
    st.subheader("Résultats sur le test set (74 104 exemples)")
    st.table(
        pd.DataFrame(
            {
                "Modèle": ["GAT", "GraphSAGE"],
                "Accuracy": ["99.71 %", "99.84 %"],
            }
        )
    )
    st.info(
        "Dataset KDD99 (1999) : les résultats ne garantissent pas les mêmes "
        "performances sur du trafic réseau actuel."
    )