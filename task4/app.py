import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="PCA Dashboard", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0b1220;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #0f1b2d;
}

[data-testid="stSidebar"] * {
    color: #e5f0ff !important;
}

h1, h2, h3 {
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

st.title("PCA Visualization & Compression Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Explained Variance",
        "2D PCA Visualization",
        "KMeans Comparison",
        "Image Reconstruction",
        "Download Model"
    ]
)

digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA().fit(X_scaled)
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)

n_components_95 = np.argmax(explained_variance >= 0.95) + 1

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

kmeans_raw = KMeans(n_clusters=10, random_state=42, n_init=10)
labels_raw = kmeans_raw.fit_predict(X_scaled)

kmeans_pca = KMeans(n_clusters=10, random_state=42, n_init=10)
labels_pca = kmeans_pca.fit_predict(X_2d)

score_raw = silhouette_score(X_scaled, labels_raw)
score_pca = silhouette_score(X_2d, labels_pca)

pca_model = PCA(n_components=n_components_95)
X_pca_opt = pca_model.fit_transform(X_scaled)

if page == "Overview":
    st.subheader("Dataset Overview")
    st.write("Digits dataset with 64 features (8x8 images).")

    fig, ax = plt.subplots()
    ax.imshow(X[0].reshape(8, 8), cmap="gray")
    ax.set_title(f"Sample Digit: {y[0]}")
    ax.axis("off")
    st.pyplot(fig)

    st.write("95% variance components:", n_components_95)

if page == "Explained Variance":
    st.subheader("Cumulative Explained Variance")

    fig, ax = plt.subplots()
    ax.plot(explained_variance)
    ax.axhline(0.95, linestyle="--")
    ax.set_xlabel("Components")
    ax.set_ylabel("Variance")
    ax.grid()
    st.pyplot(fig)

if page == "2D PCA Visualization":
    st.subheader("PCA 2D Projection")

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=10)
    plt.colorbar(scatter)
    st.pyplot(fig)

if page == "KMeans Comparison":
    st.subheader("KMeans Before vs After PCA")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_raw, cmap="viridis", s=10)
        ax.set_title("Original Data Clustering")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_pca, cmap="viridis", s=10)
        ax.set_title("PCA Reduced Clustering")
        st.pyplot(fig)

    st.write("Silhouette Score (Original):", score_raw)
    st.write("Silhouette Score (PCA):", score_pca)

if page == "Image Reconstruction":
    st.subheader("PCA Image Reconstruction")

    components_list = [10, 20, 30, 40, 50, 64]

    components_list = [c for c in components_list if c <= X.shape[1]]

    fig, axes = plt.subplots(1, len(components_list), figsize=(15, 3))

    for i, c in enumerate(components_list):
        pca = PCA(n_components=c)
        X_pca = pca.fit_transform(X_scaled)
        X_recon = pca.inverse_transform(X_pca)

        axes[i].imshow(X_recon[0].reshape(8, 8), cmap="gray")
        axes[i].set_title(f"{c} components")
        axes[i].axis("off")

    st.pyplot(fig)

if page == "Download Model":
    st.subheader("Download PCA Model")

    joblib.dump(pca_model, "pca_model.pkl")

    with open("pca_model.pkl", "rb") as f:
        st.download_button(
            "Download PCA Model (.pkl)",
            f,
            file_name="pca_model.pkl",
            mime="application/octet-stream"
        )