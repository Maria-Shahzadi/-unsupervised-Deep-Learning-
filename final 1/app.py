import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #19213c 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #f0f3fa !important;
    }
    [data-testid="stSidebar"] .stSelectbox label, 
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stFileUploader label {
        color: #cbd5e6 !important;
        font-weight: 500;
    }
    h1, h2, h3 {
        background: linear-gradient(120deg, #1e293b, #2d3a5e);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent !important;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    .stMetric {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(8px);
        border-radius: 28px;
        padding: 1rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.03);
        border: 1px solid rgba(255,255,255,0.5);
    }
    .stMetric:hover {
        transform: translateY(-2px);
        transition: 0.2s;
        border-color: #818cf8;
    }
    .stDataFrame, .stTable {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        box-shadow: 0 8px 20px rgba(79,70,229,0.3);
    }
    .stDownloadButton > button {
        background: #1e293b;
        border-radius: 40px;
    }
    .stDownloadButton > button:hover {
        background: #0f172a;
    }
    .css-1kyxreq {
        background: rgba(255,255,255,0.5);
        border-radius: 24px;
        padding: 0.5rem;
    }
    hr {
        margin: 1rem 0;
        border-color: rgba(99,102,241,0.2);
    }
    .stPlotlyChart, .stImage {
        background: rgba(255,255,255,0.5);
        border-radius: 28px;
        padding: 0.5rem;
        border: 1px solid rgba(255,255,255,0.6);
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Mall Customer Segmentation Dashboard")
st.caption("Advanced clustering analysis | KMeans | Hierarchical | DBSCAN | Beautiful insights")

uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV file", type=["csv"], help="Expects columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)")

if uploaded_file is None:
    st.info("💡 Please upload `Mall_Customers.csv` to begin the analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

page = st.sidebar.radio(
    "🧭 Navigation",
    [
        "📋 Dataset Preview",
        "📊 EDA",
        "📐 PCA Projection",
        "📉 Elbow Method",
        "🌲 Hierarchical Clustering",
        "⚡ DBSCAN Analysis",
        "🏆 Comparison & Export",
        "🔮 New Customer Prediction"
    ],
    index=0
)

data = df.copy()

if "Gender" in data.columns:
    le = LabelEncoder()
    data["Gender"] = le.fit_transform(data["Gender"].astype(str))

data = data.drop_duplicates()

num_cols = data.select_dtypes(include=np.number).columns
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

feature_df = data.copy()

if "CustomerID" in feature_df.columns:
    feature_df = feature_df.drop(columns=["CustomerID"])

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(feature_df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

agg = AgglomerativeClustering(n_clusters=5)
agg_labels = agg.fit_predict(X_scaled)

dbscan = DBSCAN(eps=1.0, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📋 Total Rows", len(df))
with col2:
    st.metric("🔢 Features Used", feature_df.shape[1])
with col3:
    st.metric("🎯 KMeans Clusters", len(np.unique(kmeans_labels)))
with col4:
    st.metric("⚠️ DBSCAN Noise Points", int(np.sum(dbscan_labels == -1)))

if page == "📋 Dataset Preview":
    st.subheader("Raw Data Overview")
    st.dataframe(df, use_container_width=True)
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(include="all"), use_container_width=True)

if page == "📊 EDA":
    st.subheader("Feature Distribution")
    numeric_columns = feature_df.select_dtypes(include=np.number).columns
    if len(numeric_columns) > 0:
        selected_feature = st.selectbox("Select feature to explore", numeric_columns)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(feature_df[selected_feature], kde=True, color="#4f46e5", ax=ax)
        ax.set_facecolor("#f9faff")
        st.pyplot(fig)
    st.subheader("Correlation Matrix")
    corr_fig, corr_ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(feature_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=corr_ax)
    st.pyplot(corr_fig)

if page == "📐 PCA Projection":
    st.subheader("2D PCA Visualization (KMeans colored)")
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(X_2d[:,0], X_2d[:,1], c=kmeans_labels, cmap="viridis", edgecolors='black', linewidth=0.3)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA reduced space")
    st.pyplot(fig)
    st.caption(f"Explained variance ratio: PC1 = {pca.explained_variance_ratio_[0]:.2%}, PC2 = {pca.explained_variance_ratio_[1]:.2%}")

if page == "📉 Elbow Method":
    st.subheader("Optimal K - Inertia Analysis")
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K_range, inertias, marker='o', linestyle='--', color="#7c3aed", linewidth=2, markersize=8)
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

if page == "🌲 Hierarchical Clustering":
    st.subheader("Dendrogram (Ward linkage)")
    linked = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linked, ax=ax, truncate_mode='lastp', p=30, leaf_rotation=45.)
    st.pyplot(fig)

if page == "⚡ DBSCAN Analysis":
    st.subheader("K‑Distance Graph (for epsilon selection)")
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X_scaled)
    distances, _ = neigh.kneighbors(X_scaled)
    k_dist = np.sort(distances[:,4])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_dist, color="#e11d48")
    ax.set_xlabel("Data points sorted")
    ax.set_ylabel("5th nearest neighbor distance")
    st.pyplot(fig)
    st.subheader("DBSCAN Clustering Result")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(X_2d[:,0], X_2d[:,1], c=dbscan_labels, cmap="tab10", edgecolors='k', linewidth=0.3)
    ax2.set_title("DBSCAN clusters (grey = noise)")
    st.pyplot(fig2)

if page == "🏆 Comparison & Export":
    valid = dbscan_labels != -1
    k_score = silhouette_score(X_scaled, kmeans_labels)
    a_score = silhouette_score(X_scaled, agg_labels)
    d_score = np.nan
    if len(np.unique(dbscan_labels[valid])) > 1:
        d_score = silhouette_score(X_scaled[valid], dbscan_labels[valid])
    comp_df = pd.DataFrame({
        "Algorithm": ["KMeans", "Agglomerative", "DBSCAN"],
        "Silhouette Score": [round(k_score,4), round(a_score,4), round(d_score,4) if not np.isnan(d_score) else "N/A"],
        "Clusters Found": [len(np.unique(kmeans_labels)), len(np.unique(agg_labels)), len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)],
        "Noise Points": [0, 0, int(np.sum(dbscan_labels == -1))]
    })
    st.subheader("Model Performance Summary")
    st.dataframe(comp_df, use_container_width=True)
    st.subheader("Visual Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(X_2d[:,0], X_2d[:,1], c=kmeans_labels, cmap="viridis", edgecolors='black', alpha=0.7)
    axes[0].set_title("KMeans")
    axes[1].scatter(X_2d[:,0], X_2d[:,1], c=agg_labels, cmap="viridis", edgecolors='black', alpha=0.7)
    axes[1].set_title("Agglomerative")
    axes[2].scatter(X_2d[:,0], X_2d[:,1], c=dbscan_labels, cmap="viridis", edgecolors='black', alpha=0.7)
    axes[2].set_title("DBSCAN")
    st.pyplot(fig)
    data["KMeans_Cluster"] = kmeans_labels
    csv_data = data.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Clustered CSV", csv_data, "clustered_customers.csv", "text/csv")

if page == "🔮 New Customer Prediction":
    st.subheader("Predict cluster for a new customer")
    col_a, col_b = st.columns(2)
    with col_a:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 18, 100, 30)
    with col_b:
        income = st.number_input("Annual Income (k$)", 1, 200, 50)
        spending = st.number_input("Spending Score (1-100)", 1, 100, 50)
    if st.button("✨ Assign Cluster", use_container_width=True):
        gender_code = 1 if gender == "Male" else 0
        new_sample = pd.DataFrame([[gender_code, age, income, spending]], columns=["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])
        new_scaled = scaler.transform(new_sample)
        pred_cluster = kmeans.predict(new_scaled)[0]
        st.success(f"✅ This customer belongs to **Cluster {pred_cluster}**")
        st.balloons()
        joblib.dump(kmeans, "kmeans_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        with open("kmeans_model.pkl", "rb") as f:
            st.download_button("💾 Download trained KMeans model", f, "kmeans_model.pkl")