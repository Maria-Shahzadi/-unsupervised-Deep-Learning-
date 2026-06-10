import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Clustering Lab", layout="wide")

st.markdown("""
<style>

.main {
    background-color: #0b1220;
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #0f1b2d;
    border-right: 2px solid #1f3b57;
}

[data-testid="stSidebar"] * {
    color: #e5f0ff !important;
}

section[data-testid="stSidebar"] {
    padding-top: 20px;
}

.css-1d391kg {
    background-color: #0f1b2d;
}

.stSelectbox, .stRadio {
    background-color: #111c2e;
    border-radius: 10px;
    padding: 10px;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
}

h1, h2, h3 {
    color: #f8fafc;
}

</style>
""", unsafe_allow_html=True)

st.title("Clustering on Complex Shapes")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Overview",
        "K-Distance Graph",
        "Cluster Visualization",
        "Silhouette Scores",
        "Comparison Table",
        "Conclusion"
    ]
)

moons_X, _ = make_moons(n_samples=500, noise=0.08, random_state=42)
circles_X, _ = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)

datasets = {
    "Moons": moons_X,
    "Circles": circles_X
}

def get_models(X):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    agg = AgglomerativeClustering(n_clusters=2)
    dbscan = DBSCAN(eps=0.2, min_samples=5)

    return {
        "K-Means": kmeans.fit_predict(X),
        "Agglomerative": agg.fit_predict(X),
        "DBSCAN": dbscan.fit_predict(X)
    }

def silhouette_results(X, labels_dict):
    res = []
    for name, labels in labels_dict.items():
        if len(np.unique(labels)) > 1:
            if name == "DBSCAN":
                mask = labels != -1
                if len(np.unique(labels[mask])) > 1:
                    score = silhouette_score(X[mask], labels[mask])
                    res.append([name, score])
            else:
                score = silhouette_score(X, labels)
                res.append([name, score])
    return pd.DataFrame(res, columns=["Algorithm", "Silhouette Score"])

selected_dataset = st.selectbox("Select Dataset", list(datasets.keys()))
X = datasets[selected_dataset]

labels_dict = get_models(X)

if page == "Dataset Overview":
    st.subheader("Dataset Shape View")
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=15, c="#60a5fa")
    st.pyplot(fig)

if page == "K-Distance Graph":
    st.subheader("K-Distance Graph for DBSCAN Epsilon Selection")
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    distances = np.sort(distances[:, 4])

    fig, ax = plt.subplots()
    ax.plot(distances, color="#38bdf8")
    ax.set_xlabel("Points")
    ax.set_ylabel("5-NN Distance")
    ax.grid(True, color="#334155")
    st.pyplot(fig)

if page == "Cluster Visualization":
    st.subheader("Clustering Comparison")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ["#60a5fa", "#fbbf24", "#34d399"]

    for ax, (name, labels) in zip(axes, labels_dict.items()):
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20)
        ax.set_title(name, color="white")
        ax.set_facecolor("#0f1b2d")

    st.pyplot(fig)

if page == "Silhouette Scores":
    st.subheader("Silhouette Score Comparison")
    df = silhouette_results(X, labels_dict)
    st.dataframe(df, use_container_width=True)

if page == "Comparison Table":
    st.subheader("Dataset Comparison")

    all_results = []

    for name, data in datasets.items():
        models = get_models(data)
        df = silhouette_results(data, models)
        df["Dataset"] = name
        all_results.append(df)

    final_df = pd.concat(all_results)
    pivot = final_df.pivot(index="Dataset", columns="Algorithm", values="Silhouette Score")

    st.dataframe(pivot, use_container_width=True)

if page == "Conclusion":
    st.subheader("Final Conclusion")

    st.markdown("""
    K-Means works well for simple convex structures but fails on complex shapes.  
    Agglomerative clustering performs moderately well.  
    DBSCAN gives the best results for non-linear datasets like moons and circles because it detects density-based clusters and noise effectively.
    """)