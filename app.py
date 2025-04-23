import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import warnings

# Hide specific warnings
warnings.filterwarnings("ignore")

# Load models
@st.cache_resource
def load_models():
    yolo = YOLO('yolo_best.pt')  # Update path as needed
    effnet = tf.keras.models.load_model('best_apple_model.h5')
    return yolo, effnet

yolo_model, efficientnet_model = load_models()
class_id_map = [81, 82, 83, 84, 85]

# --- Custom CSS for Tailwind-style vibes ---
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f9fafb;
    }
    .stApp {
        max-width: 1200px;
        margin: auto;
        padding: 2rem;
    }
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 600;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        border-radius: 0.5rem;
        background-color: #6a0dad;
        color: white;
        padding: 0.5rem 1rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stImage>div>img {
        border-radius: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .ripening-note {
            background-color: #f9f9f9;
            padding: 1rem;
            border-radius: 12px;
            font-size: 16px;
            line-height: 1.6;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        .ripening-note span {
            font-weight: bold;
            color: #6a0dad;
</style>


""", unsafe_allow_html=True)

# --- Title and uploader ---
st.markdown("## 🍎 Apple Detector & Classifier")
st.write("Upload an image to detect and classify apples using YOLO + EfficientNet.")
with st.container():
    st.markdown("""
<div class="ripening-note">
        <p><span>85</span> → Advanced ripening (almost mature / fully colored)</p>
        <p><span>84</span> → Ripe stage</p>
        <p><span>83</span> → Sugar level rises significantly</p>
        <p><span>82</span> → Starch is rapidly breaking down into simple sugars</p>
        <p><span>81</span> → Beginning of ripening (start of color change)</p>
    </div>

    """, unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize large images
    max_dim = 1024
    if image.width > max_dim or image.height > max_dim:
        image.thumbnail((max_dim, max_dim))

    image_np = np.array(image)
    st.image(image, caption='📷 Resized Input Image', use_container_width=True)

    # Run detection
    with st.spinner("🔍 Detecting apples with YOLO..."):
        results = yolo_model(image_np)
    detections = results[0].boxes.xyxy.cpu().numpy()

    if len(detections) == 0:
        st.warning("No apples detected in the image.")
    else:
        st.markdown("### 🍏 Cropped Apples & Predictions")
        cols = st.columns(3)

        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box)
            cropped_img = image_np[y1:y2, x1:x2]

            # Preprocess for EfficientNet
            resized_img = cv2.resize(cropped_img, (224, 224))
            input_tensor = tf.keras.applications.efficientnet.preprocess_input(resized_img)
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # Predict
            with st.spinner("🧠 Classifying..."):
                prediction = efficientnet_model.predict(input_tensor)
            class_index = np.argmax(prediction)
            class_id = class_id_map[class_index]
            confidence = np.max(prediction)

            # Display side-by-side
            col = cols[i % len(cols)]
            with col:
                st.image(cropped_img, caption=f"🍎 ID: {class_id} ({confidence:.2f})", use_container_width=True)
                time.sleep(0.3)
