import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_digits, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="ANN Dashboard", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.main {
    background-color: #0f172a;
}
.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 10px;
    height: 45px;
    width: 100%;
}
.stSelectbox {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("ANN Dashboard (Sklearn + Scratch)")

menu = st.sidebar.selectbox("Choose Module", ["Sklearn ANN", "Scratch ANN", "Comparison"])

if menu == "Sklearn ANN":

    @st.cache_data
    def load_data():
        data = load_digits()
        X = data.data / 16.0
        y = data.target
        return train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_test, y_train, y_test = load_data()

    if st.button("Train ANN Model"):

        model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=15)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        st.success(f"Accuracy: {acc}")

        fig = plt.figure()
        plt.plot(model.loss_curve_)
        plt.title("Loss Curve")
        st.pyplot(fig)

        joblib.dump(model, "ann_model.pkl")

    if st.button("Download Model"):
        with open("ann_model.pkl", "rb") as f:
            st.download_button("Download Model", f, file_name="ann_model.pkl")


if menu == "Scratch ANN":

    X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def forward(X, W1, b1, W2, b2):
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        return z1, a1, z2, a2

    def loss(y, yhat):
        return -np.mean(y*np.log(yhat+1e-8) + (1-y)*np.log(1-yhat+1e-8))

    W1 = np.random.randn(2, 8) * 0.1
    b1 = np.zeros((1, 8))
    W2 = np.random.randn(8, 1) * 0.1
    b2 = np.zeros((1, 1))

    if st.button("Train Scratch ANN"):

        losses = []

        for i in range(1500):

            z1, a1, z2, a2 = forward(X_train, W1, b1, W2, b2)
            l = loss(y_train, a2)
            losses.append(l)

            dz2 = a2 - y_train
            dW2 = a1.T @ dz2 / len(X_train)
            db2 = np.sum(dz2, axis=0, keepdims=True) / len(X_train)

            dz1 = (dz2 @ W2.T) * a1 * (1 - a1)
            dW1 = X_train.T @ dz1 / len(X_train)
            db1 = np.sum(dz1, axis=0, keepdims=True) / len(X_train)

            W1 -= 0.1 * dW1
            b1 -= 0.1 * db1
            W2 -= 0.1 * dW2
            b2 -= 0.1 * db2

        fig = plt.figure()
        plt.plot(losses)
        plt.title("Loss Curve")
        st.pyplot(fig)

        _, _, _, pred = forward(X_test, W1, b1, W2, b2)
        pred = (pred > 0.5).astype(int)

        st.success(f"Accuracy: {np.mean(pred == y_test)}")

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        grid = np.c_[xx.ravel(), yy.ravel()]
        _, _, _, a2 = forward(grid, W1, b1, W2, b2)
        Z = a2.reshape(xx.shape)

        fig2 = plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.5)
        plt.scatter(X[:, 0], X[:, 1], c=y.ravel())
        plt.title("Decision Boundary")
        st.pyplot(fig2)


if menu == "Comparison":

    st.subheader("Model Comparison")

    df = pd.DataFrame({
        "Model": ["Sklearn ANN (Digits)", "Scratch ANN (Moons)"],
        "Status": ["Ready", "Ready"]
    })

    st.dataframe(df)