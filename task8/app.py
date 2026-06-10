import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image

st.set_page_config(page_title="CNN MNIST Dashboard", layout="wide")

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

st.title("CNN MNIST Prediction Dashboard")

model = keras.models.load_model("cnn_mnist.keras")

menu = st.sidebar.selectbox("Select Option", ["Predict Digit", "Show Feature Maps"])

# ---------------- PREDICTION ----------------
if menu == "Predict Digit":

    st.subheader("Upload Digit Image (28x28)")

    file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if file is not None:

        img = Image.open(file).convert("L")
        img = img.resize((28,28))

        st.image(img, caption="Input Image", width=150)

        img = np.array(img)/255.0
        img = img.reshape(1,28,28,1)

        pred = model.predict(img)
        label = np.argmax(pred)

        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg,#ff416c,#ff4b2b);
            padding:20px;
            border-radius:15px;
            text-align:center;
            font-size:28px;
            font-weight:bold;">
            Predicted Digit: {label}
        </div>
        """, unsafe_allow_html=True)

# ---------------- FEATURE MAPS ----------------
if menu == "Show Feature Maps":

    st.subheader("CNN Feature Maps Visualization")

    from tensorflow.keras.models import Model

    conv_layers = [layer.output for layer in model.layers if 'conv' in layer.name]
    activation_model = Model(inputs=model.input, outputs=conv_layers)

    file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if file is not None:

        img = Image.open(file).convert("L")
        img = img.resize((28,28))
        img = np.array(img)/255.0
        img = img.reshape(1,28,28,1)

        activations = activation_model.predict(img)

        st.write("First Conv Layer Feature Maps")

        fig, axes = plt.subplots(2,4, figsize=(10,5))

        for i in range(8):
            axes[i//4, i%4].imshow(activations[0][0,:,:,i], cmap='viridis')
            axes[i//4, i%4].axis('off')

        st.pyplot(fig)