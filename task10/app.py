import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="Fashion-MNIST CNN Dashboard",
    page_icon="👕",
    layout="wide"
)

class_names = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fashion_mnist_cnn.h5")

model = load_model()

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.block-container {
    padding-top: 1rem;
}
.metric-container {
    border-radius: 12px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Fashion-MNIST CNN")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Dataset",
        "Model Architecture",
        "Image Prediction",
        "Class Probabilities"
    ]
)

if page == "Project Overview":

    st.title("End-to-End Image Classification")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Classes", "10")

    with c2:
        st.metric("Image Size", "28 x 28")

    with c3:
        st.metric("Model", "CNN")

    st.markdown("---")

    st.subheader("Dataset")

    st.write("""
Fashion-MNIST contains 10 categories of clothing images.
Each image is grayscale with a size of 28×28 pixels.
The model classifies uploaded images into one of the 10 categories.
""")

    st.subheader("Features")

    st.write("""
    • Data Augmentation

    • Batch Normalization

    • Dropout Regularization

    • Adam Optimizer

    • Early Stopping

    • Reduce Learning Rate on Plateau

    • CNN with Three Convolution Blocks
    """)

elif page == "Dataset":

    st.title("Dataset Classes")

    data = pd.DataFrame(
        {
            "Class ID": list(range(10)),
            "Class Name": class_names
        }
    )

    st.dataframe(
        data,
        use_container_width=True
    )

elif page == "Model Architecture":

    st.title("CNN Architecture")

    architecture = pd.DataFrame(
        [
            ["Input", "28x28x1"],
            ["Conv2D", "32 Filters"],
            ["BatchNormalization", "-"],
            ["MaxPooling", "-"],
            ["Conv2D", "64 Filters"],
            ["BatchNormalization", "-"],
            ["MaxPooling", "-"],
            ["Conv2D", "128 Filters"],
            ["BatchNormalization", "-"],
            ["MaxPooling", "-"],
            ["Flatten", "-"],
            ["Dense", "256"],
            ["Dropout", "0.5"],
            ["Dense", "128"],
            ["Dropout", "0.3"],
            ["Output", "10 Classes"]
        ],
        columns=["Layer", "Details"]
    )

    st.dataframe(
        architecture,
        use_container_width=True
    )

elif page == "Image Prediction":

    st.title("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width=250
        )

        gray = image.convert("L")
        resized = gray.resize((28, 28))

        img_array = np.array(resized)

        img_array = img_array.astype("float32") / 255.0

        img_array = np.expand_dims(img_array, axis=-1)

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        predicted_class = np.argmax(prediction)

        confidence = float(np.max(prediction))

        st.success(
            f"Prediction: {class_names[predicted_class]}"
        )

        st.info(
            f"Confidence: {confidence:.2%}"
        )

elif page == "Class Probabilities":

    st.title("Prediction Probability Analysis")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"],
        key="probability"
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            width=250
        )

        gray = image.convert("L")
        resized = gray.resize((28, 28))

        img_array = np.array(resized)

        img_array = img_array.astype("float32") / 255.0

        img_array = np.expand_dims(img_array, axis=-1)

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]

        probability_df = pd.DataFrame(
            {
                "Class": class_names,
                "Probability": prediction
            }
        )

        st.dataframe(
            probability_df,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            class_names,
            prediction
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)