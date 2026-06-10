import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

st.set_page_config(
    page_title="Hierarchical Clustering Dashboard",
    page_icon="🌸",
    layout="wide"
)

st.markdown("""
<style>

.main{
    background-color:#f8fafc;
}

[data-testid="stSidebar"]{
    background:#1e40af;
}

[data-testid="stSidebar"] *{
    color:white;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1e3a8a;
    margin-bottom:10px;
}

.subtitle{
    text-align:center;
    color:#64748b;
    font-size:18px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='title'>🌸 Hierarchical Clustering Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Agglomerative Clustering using Iris Dataset</div>",
    unsafe_allow_html=True
)

st.write("")

st.sidebar.header("⚙️ Settings")

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    2,
    6,
    3
)

run_button = st.sidebar.button(
    "🚀 Run Analysis"
)

iris = load_iris()

X = iris.data[:, :2]

feature_names = iris.feature_names[:2]

df = pd.DataFrame(
    X,
    columns=feature_names
)

if run_button:

    st.success("Analysis Completed Successfully!")

    st.subheader("📄 Iris Dataset Preview")

    st.dataframe(df.head(20))

    st.divider()

    st.subheader("🌳 Dendrograms")

    linkage_methods = [
        "single",
        "complete",
        "average",
        "ward"
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14,10)
    )

    axes = axes.ravel()

    for i, method in enumerate(linkage_methods):

        Z = linkage(
            X,
            method=method
        )

        dendrogram(
            Z,
            ax=axes[i]
        )

        axes[i].set_title(
            f"{method.capitalize()} Linkage"
        )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    st.subheader("📊 Cluster Visualization")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14,10)
    )

    axes = axes.ravel()

    results = []

    models = {}

    for i, method in enumerate(linkage_methods):

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=method
        )

        labels = model.fit_predict(X)

        models[method] = model

        score = silhouette_score(
            X,
            labels
        )

        results.append(
            [method, score]
        )

        axes[i].scatter(
            X[:,0],
            X[:,1],
            c=labels,
            cmap="viridis",
            s=50
        )

        axes[i].set_title(
            f"{method.capitalize()}\nSilhouette={score:.3f}"
        )

        axes[i].set_xlabel(
            feature_names[0]
        )

        axes[i].set_ylabel(
            feature_names[1]
        )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    st.subheader("🏆 Silhouette Score Comparison")

    comparison_df = pd.DataFrame(
        results,
        columns=[
            "Linkage Method",
            "Silhouette Score"
        ]
    )

    comparison_df = comparison_df.sort_values(
        by="Silhouette Score",
        ascending=False
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    best_method = comparison_df.iloc[0]["Linkage Method"]

    best_score = comparison_df.iloc[0]["Silhouette Score"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Best Linkage Method",
            best_method.capitalize()
        )

    with col2:
        st.metric(
            "Best Silhouette Score",
            round(best_score, 4)
        )

    st.divider()

    st.subheader("📈 Silhouette Score Chart")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        comparison_df["Linkage Method"],
        comparison_df["Silhouette Score"]
    )

    ax.set_title(
        "Linkage Method Comparison"
    )

    ax.set_ylabel(
        "Silhouette Score"
    )

    st.pyplot(fig)

    st.divider()

    st.subheader("📝 Observation")

    st.info(
        f"""
        The best performing linkage method is
        {best_method.capitalize()}
        with a Silhouette Score of
        {best_score:.4f}.

        Single linkage generally produces elongated clusters due to the chaining effect.

        Complete linkage forms compact clusters.

        Average linkage provides balanced clustering.

        Ward linkage minimizes within-cluster variance and often produces the most compact and well-separated clusters on the Iris dataset.
        """
    )

    best_model = models[best_method]

    joblib.dump(
        best_model,
        "iris_hierarchical_model.pkl"
    )

    st.subheader("💾 Download Best Model")

    with open(
        "iris_hierarchical_model.pkl",
        "rb"
    ) as file:

        st.download_button(
            label="⬇ Download .pkl Model",
            data=file,
            file_name="iris_hierarchical_model.pkl",
            mime="application/octet-stream"
        )

else:

    st.info(
        "Select parameters from the sidebar and click 'Run Analysis'."
    )