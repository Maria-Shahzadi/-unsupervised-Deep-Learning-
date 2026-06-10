import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Perceptron From Scratch", layout="wide")

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=20):
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.errors = []
        self.history = []

        for _ in range(self.epochs):
            errors = 0

            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                prediction = self.activation(linear_output)

                update = self.lr * (target - prediction)

                self.weights += update * xi
                self.bias += update

                if update != 0:
                    errors += 1

            self.errors.append(errors)
            self.history.append((self.weights.copy(), self.bias))

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)


def plot_boundary(X, y, weights, bias, title):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=180)

    x1 = np.linspace(-0.5, 1.5, 100)

    if len(weights) > 1 and weights[1] != 0:
        x2 = -(weights[0] * x1 + bias) / weights[1]
        ax.plot(x1, x2)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(title)
    ax.grid(True)

    return fig


def plot_errors(errors):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(range(1, len(errors) + 1), errors, marker="o")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassifications")
    ax.set_title("Error Curve")
    ax.grid(True)

    return fig


st.title("Perceptron From Scratch")

dataset = st.sidebar.selectbox(
    "Dataset",
    ["AND Gate", "OR Gate", "XOR Gate"]
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    0.01,
    1.0,
    0.1
)

epochs = st.sidebar.slider(
    "Epochs",
    1,
    100,
    20
)

if dataset == "AND Gate":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

elif dataset == "OR Gate":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

else:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

if st.button("Train Model"):
    model = Perceptron(
        learning_rate=learning_rate,
        epochs=epochs
    )

    model.fit(X, y)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parameters")
        st.write("Weights:", model.weights)
        st.write("Bias:", model.bias)

        predictions = model.predict(X)

        st.subheader("Predictions")
        st.write(predictions)

    with col2:
        st.subheader("Dataset")

        st.dataframe({
            "X1": X[:, 0],
            "X2": X[:, 1],
            "Target": y
        })

    st.subheader("Decision Boundary")
    st.pyplot(
        plot_boundary(
            X,
            y,
            model.weights,
            model.bias,
            dataset
        )
    )

    st.subheader("Error Curve")
    st.pyplot(plot_errors(model.errors))

    st.subheader("Decision Boundary Updates")

    for epoch, (w, b) in enumerate(model.history):
        st.pyplot(
            plot_boundary(
                X,
                y,
                w,
                b,
                f"Epoch {epoch + 1}"
            )
        )

    model_data = {
        "weights": model.weights,
        "bias": model.bias
    }

    st.download_button(
        "Download Model",
        data=pickle.dumps(model_data),
        file_name=f"{dataset.replace(' ', '_')}.pkl",
        mime="application/octet-stream"
    )

    if dataset == "XOR Gate":
        st.subheader("Why XOR Fails")
        st.write(
            """
            XOR is not linearly separable.

            A single-layer perceptron can learn only one linear decision boundary.

            Therefore, it cannot classify XOR correctly.

            A Multi-Layer Perceptron with one or more hidden layers is required.
            """
        )

st.markdown("---")

st.header("Theory")

st.write(
    """
    AND and OR datasets are linearly separable, so a perceptron converges successfully.

    XOR is not linearly separable because no single straight line can separate its classes.

    The architectural change required to solve XOR is a Multi-Layer Perceptron with hidden layers and nonlinear activation functions.
    """
)
