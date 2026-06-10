import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os



st.set_page_config(
    page_title="K-Means Clustering Dashboard",
    page_icon="📊",
    layout="wide"
)


st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

[data-testid="stSidebar"]{
    background: #1e40af;
}

[data-testid="stSidebar"] *{
    color:white;
}

.dashboard-title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#1e3a8a;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}

.metric-box{
    background:#dbeafe;
    padding:15px;
    border-radius:12px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)



st.markdown(
    "<div class='dashboard-title'>📊 K-Means Clustering Dashboard</div>",
    unsafe_allow_html=True
)

st.write("")


st.sidebar.header("⚙️ Settings")

n_samples = st.sidebar.slider(
    "Number of Samples",
    100,
    1000,
    300
)

n_clusters = st.sidebar.slider(
    "Number of Clusters (K)",
    2,
    10,
    3
)

cluster_std = st.sidebar.slider(
    "Cluster Standard Deviation",
    0.5,
    3.0,
    1.0
)

run_button = st.sidebar.button("🚀 Run Clustering")


class KMeansScratch:

    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):

        np.random.seed(42)

        random_idx = np.random.choice(
            len(X),
            self.k,
            replace=False
        )

        self.centroids = X[random_idx]

        for _ in range(self.max_iters):

            distances = np.linalg.norm(
                X[:, np.newaxis] - self.centroids,
                axis=2
            )

            labels = np.argmin(
                distances,
                axis=1
            )

            new_centroids = np.array([
                X[labels == i].mean(axis=0)
                for i in range(self.k)
            ])

            if np.allclose(
                self.centroids,
                new_centroids
            ):
                break

            self.centroids = new_centroids

        self.labels_ = labels



if run_button:

    # Dataset

    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=42
    )

    
    manual_model = KMeansScratch(k=n_clusters)

    manual_model.fit(X)

    manual_labels = manual_model.labels_

    manual_centroids = manual_model.centroids

    # Sklearn KMeans

    sklearn_model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    sklearn_model.fit(X)

    sklearn_labels = sklearn_model.labels_

    sklearn_centroids = sklearn_model.cluster_centers_

    # Save Model

    joblib.dump(
        sklearn_model,
        "kmeans_model.pkl"
    )

    # Silhouette Scores

    manual_score = silhouette_score(
        X,
        manual_labels
    )

    sklearn_score = silhouette_score(
        X,
        sklearn_labels
    )


    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Manual Silhouette Score",
            round(manual_score, 4)
        )

    with col2:
        st.metric(
            "Sklearn Silhouette Score",
            round(sklearn_score, 4)
        )

    st.divider()


    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Manual K-Means")

        fig, ax = plt.subplots(figsize=(6,5))

        ax.scatter(
            X[:,0],
            X[:,1],
            c=manual_labels,
            cmap="viridis"
        )

        ax.scatter(
            manual_centroids[:,0],
            manual_centroids[:,1],
            marker="X",
            s=300,
            color="red"
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Sklearn K-Means")

        fig, ax = plt.subplots(figsize=(6,5))

        ax.scatter(
            X[:,0],
            X[:,1],
            c=sklearn_labels,
            cmap="viridis"
        )

        ax.scatter(
            sklearn_centroids[:,0],
            sklearn_centroids[:,1],
            marker="X",
            s=300,
            color="red"
        )

        st.pyplot(fig)

    st.divider()


    st.subheader("📈 Elbow Method")

    wcss = []

    for k in range(1,11):

        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        km.fit(X)

        wcss.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(
        range(1,11),
        wcss,
        marker='o'
    )

    ax.set_xlabel("K")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Curve")

    st.pyplot(fig)

    st.divider()



    st.subheader("📌 Centroid Comparison")

    manual_df = pd.DataFrame(
        manual_centroids,
        columns=["X","Y"]
    )

    sklearn_df = pd.DataFrame(
        sklearn_centroids,
        columns=["X","Y"]
    )

    c1, c2 = st.columns(2)

    with c1:
        st.write("Manual Centroids")
        st.dataframe(manual_df)

    with c2:
        st.write("Sklearn Centroids")
        st.dataframe(sklearn_df)

    st.divider()

    st.subheader("📄 Dataset Preview")

    df = pd.DataFrame(
        X,
        columns=["Feature 1","Feature 2"]
    )

    st.dataframe(df.head(20))

    st.divider()


    st.subheader("💾 Download Trained Model")

    with open(
        "kmeans_model.pkl",
        "rb"
    ) as file:

        st.download_button(
            label="⬇ Download Model",
            data=file,
            file_name="kmeans_model.pkl",
            mime="application/octet-stream"
        )

else:

    st.info(
        "Select parameters from the sidebar and click 'Run Clustering'."
    )