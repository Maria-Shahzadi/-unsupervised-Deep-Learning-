import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

st.set_page_config(page_title="FNN Dashboard", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0b1020;
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border-radius: 10px;
    width: 100%;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("Feedforward Neural Network Dashboard")

menu = st.sidebar.selectbox("Select", ["Train Model", "Predict Sample", "Upload Image"])

model_path = "fnn_model.pkl"

data = load_digits()
X = data.data / 16.0
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if menu == "Train Model":

    st.subheader("Train Feedforward Neural Network")

    if st.button("Start Training"):

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            max_iter=20
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        st.success(f"Accuracy: {acc}")

        plt.figure()
        plt.plot(model.loss_curve_)
        plt.title("Loss Curve")
        st.pyplot(plt)

        joblib.dump(model, model_path)
        st.success("Model Saved Successfully")

if menu == "Predict Sample":

    st.subheader("Test Random Samples")

    if st.button("Load Model & Predict"):

        model = joblib.load(model_path)

        idx = np.random.randint(0, len(X_test), 5)

        preds = model.predict(X_test[idx])

        st.write("Predictions:", preds)
        st.write("Actual:", y_test[idx])

        fig, ax = plt.subplots(1, 5, figsize=(10, 3))

        for i, a in enumerate(ax):
            a.imshow(X_test[idx[i]].reshape(8, 8), cmap='gray')
            a.set_title(f"P:{preds[i]}\nT:{y_test[idx[i]]}")
            a.axis("off")

        st.pyplot(fig)

if menu == "Upload Image":

    st.subheader("Upload Digit Image")

    file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if file is not None:

        model = joblib.load(model_path)

        img = Image.open(file).convert("L")
        img = img.resize((8, 8))

        st.image(img, width=150)

        img = np.array(img)

        img = 16 - (img / 16.0 * 16)
        img = img.reshape(1, -1)

        pred = model.predict(img)

        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg,#ff416c,#ff4b2b);
            padding:20px;
            border-radius:15px;
            text-align:center;
            font-size:28px;
            font-weight:bold;">
            Predicted Digit: {pred[0]}
        </div>
        """, unsafe_allow_html=True)