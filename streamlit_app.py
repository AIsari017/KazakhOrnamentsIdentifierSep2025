import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ----- CONFIG -----
IMAGE_SIZE = (224, 224)  
MODEL_PATH = "my_image_classifier_model_v1.keras"
CLASS_NAMES_PATH = "class_names.json"

# ----- PAGE -----
st.set_page_config(page_title="Ornament Classifier")
st.title("Ornament Classifier")

# ----- HELPERS -----
@st.cache_resource(show_spinner=True)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_class_names():
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None  # fall back to numeric labels

def preprocess(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    x = tf.keras.utils.img_to_array(img)

    # IMPORTANT: use the SAME preprocess you trained with
    # If you trained with EfficientNetB0:
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    # shape -> (1, H, W, 3)
    return np.expand_dims(x, axis=0)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# ----- UI -----
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    example = st.checkbox("Use example image")
if example:
    st.info("Add an example image file if you like, or uncheck this.")
    # fall back to uploader-only demo

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded", use_container_width=True)

    with st.spinner("Loading model & predicting..."):
        model = load_model()
        class_names = load_class_names()

        x = preprocess(image)
        preds = model.predict(x, verbose=0)  
        if preds.shape[-1] > 1 and preds.max() <= 1.0:
            probs = preds[0]
        else:
            probs = softmax(preds)[0]

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        label = class_names[top_idx] if class_names else f"class_{top_idx}"

    st.subheader(f"Prediction: **{label}** ({top_prob:.2%})")
    st.progress(top_prob)

    # Show full distribution
    if class_names:
        st.write("Class probabilities:")
        for i, p in sorted(enumerate(probs), key=lambda t: -t[1]):
            st.write(f"- {class_names[i]}: {p:.2%}")
