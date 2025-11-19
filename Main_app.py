import os
import sys
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import io
import matplotlib.pyplot as plt
import pandas as pd

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def load_css(css_file: str = "styles.css"):
    path = resource_path(css_file)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<style>"
            "/* styles.css not found — using default look */"
            ".stApp{background-color:#101820 !important;}"
            "</style>",
            unsafe_allow_html=True,
        )

st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="icon3.jpeg",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.logo("icon3.jpeg", size = "large", icon_image="icon3.jpeg")

load_css("styles.css")

@st.cache_resource
def load_tumor_model(path="best_model.h5"):
    return tf.keras.models.load_model(path, compile=False)

try:
    model = load_tumor_model(resource_path("best_model.h5"))
except Exception as e:
    st.error("Error loading model. Make sure `best_model.h5` is present and compatible with your TensorFlow version.")
    st.exception(e)
    st.stop()

CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

logo_path = resource_path("assets/bt_128.png")
if os.path.exists(logo_path):
    try:
        st.image(logo_path, width=96)
    except Exception:
        pass

st.title("Brain Tumor Detector")
st.caption("Simple desktop app for predicting the type of brain tumor if it is present. It is strictly for educational use, not a medical device.")

with st.sidebar:
    st.header("How to use")
    st.write("""
    1. Upload one or more MRI images (JPG/JPEG/PNG).  
    2. Click **Predict**.  
    """)
    st.markdown("---")
    st.header("Options")
    show_probs = st.checkbox("Show full probability bar", value=False)
    st.markdown("---")
    st.caption("Model: MobileNetV2 (transfer learning).")

def preprocess_pil(img: Image.Image, size=(224,224)):
    img = img.convert("RGB").resize(size)
    arr = img_to_array(img) / 255.0
    return arr

def predict_image(img: Image.Image):
    x = np.expand_dims(preprocess_pil(img), axis=0)
    preds = model.predict(x)
    if preds is None:
        return None, None
    preds = np.asarray(preds)[0]
    top_idx = int(np.argmax(preds))
    return preds, top_idx

uploaded = st.file_uploader("Upload MRI images (JPG/JPEG/PNG)", accept_multiple_files=True, type=["jpg","jpeg","png"])

if uploaded:
    if st.button("Predict"):
        results = []
        cols = st.columns(2)

        for i, file in enumerate(uploaded):
            col = cols[i % 2]
            with col:
                try:
                    img = Image.open(file)
                except Exception:
                    st.error(f"Cannot open {getattr(file, 'name', 'uploaded file')}")
                    continue

                st.image(img, caption=getattr(file, 'name', ''), use_container_width=True)

                preds, top_idx = predict_image(img)
                if preds is None:
                    st.error("Prediction failed for this image.")
                    continue

                pred_label = CLASSES[top_idx]
                conf = float(preds[top_idx])

                if conf < 0.60:
                    st.warning(f"Uncertain prediction (low confidence: {conf:.2%}).")
                    final_label = "Uncertain"
                else:
                    final_label = pred_label.capitalize()
                    st.success(f"Prediction: **{final_label}** ({conf:.2%})")

                if show_probs:
                    probs_df = {CLASSES[j]: float(preds[j]) for j in range(len(CLASSES))}
                    st.bar_chart(pd.DataFrame([probs_df]))

                results.append({
                    "Image": getattr(file, 'name', f'image_{i}'),
                    "Prediction": final_label,
                    "Confidence": f"{conf:.2%}"
                })

        
        if results:
            df = pd.DataFrame(results)
            st.table(df)

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Predictions")
            buf.seek(0)

            st.download_button(
                "Download Excel Report",
                data=buf,
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("Upload MRIs to get predictions.")

