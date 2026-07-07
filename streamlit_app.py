"""
streamlit_app.py
-----------------
Streamlit web application for the Flower Classifier Deep Learning project.

Lets a user upload a flower image in the browser and get a real-time
prediction from the trained CNN model.

Run with:
    streamlit run streamlit_app.py

Requirements (add to requirements.txt if not already present):
    streamlit, tensorflow, numpy, Pillow
"""

import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Configuration — update these to match your trained model
# ---------------------------------------------------------------------------
MODEL_PATH = "models/flower_classifier_model.h5"
IMAGE_SIZE = (150, 150)  # must match the size used during training

# Order must match train_generator.class_indices from the notebook
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

EMOJI = {
    "rose": "🌹",
    "sunflower": "🌻",
    "tulip": "🌷",
    "daisy": "🌼",
    "dandelion": "🌺",
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered",
)

st.title("🌸 Flower Classifier")
st.write(
    "Upload a flower image and this CNN model will predict its species "
    "(rose, sunflower, tulip, daisy, or dandelion)."
)


@st.cache_resource
def get_model(model_path):
    """Load the Keras model once and cache it across reruns."""
    return load_model(model_path)


def preprocess_image(pil_img, target_size=IMAGE_SIZE):
    """Resize, normalize, and batch a PIL image for model input."""
    img = pil_img.convert("RGB").resize(target_size)
    img_array = np.array(img, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(model, pil_img):
    img_array = preprocess_image(pil_img)
    predictions = model.predict(img_array, verbose=0)[0]
    top_index = int(np.argmax(predictions))
    label = CLASS_NAMES[top_index]
    confidence = float(predictions[top_index]) * 100
    return label, confidence, predictions


# ---------------------------------------------------------------------------
# Main app logic
# ---------------------------------------------------------------------------
if not os.path.isfile(MODEL_PATH):
    st.error(
        f"Model file not found at `{MODEL_PATH}`.\n\n"
        "Train the model in `Flower_Classifier.ipynb` and save it to this "
        "path, or update `MODEL_PATH` at the top of this script."
    )
    st.stop()

model = get_model(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Choose a flower image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        label, confidence, predictions = predict(model, pil_img)

    emoji = EMOJI.get(label, "🌸")
    st.success(f"Predicted Flower Species: {emoji} **{label.capitalize()}**")
    st.metric("Confidence Score", f"{confidence:.2f}%")

    st.subheader("Class Probabilities")
    prob_data = {
        name: float(prob) for name, prob in zip(CLASS_NAMES, predictions)
    }
    st.bar_chart(prob_data)
else:
    st.info("👆 Upload an image to get started.")

st.markdown("---")
st.caption("Flower Classifier Deep Learning project — CNN-powered flower recognition.")
