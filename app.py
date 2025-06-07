import streamlit as st
from ultralytics import YOLO
import torch
from PIL import Image
import requests
import os

# Function to download model weights from Google Drive
def download_file_from_google_drive(id, destination):
    if not os.path.exists(destination):
        url = f"https://drive.google.com/uc?id={id}"
        response = requests.get(url, allow_redirects=True)
        with open(destination, 'wb') as f:
            f.write(response.content)

# Download both models from Google Drive
download_file_from_google_drive("1110aAQtjWpViRoSOI2B9s6eS7uFCaz7M", "clean_reusable_model.pt")
download_file_from_google_drive("1zLG_0gNfV59YxEyovad0aU79rTnYylby", "reusable_vs_nonreusable.pt")

# Load both models using YOLOv8
@st.cache_resource
def load_model(path):
    return YOLO(path)

model1 = load_model("reusable_vs_nonreusable.pt")
model2 = load_model("clean_reusable_model.pt")

# Inference explanation logic
def get_explanation(label, confidence):
    confidence_pct = round(confidence * 100, 2)

    if label == "nonreusable":
        return f"🟥 Nonreusable detected with {confidence_pct}% confidence. This may be due to:\n- Dark brown/black oil color\n- Floating food particles\n- Cloudy or sticky texture\n- Irregular reflections or dirty vessel."
    elif label == "reusable":
        return f"🟩 Reusable detected with {confidence_pct}% confidence. Indicators include:\n- Clear golden oil\n- No visible food particles\n- Clean container and fresh look."
    else:
        return "⚠️ No reusable oil detected in the image."

# Streamlit UI
st.title("🛢️ Reusable Oil Detector")
st.write("Upload an image and choose a model to check oil quality.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("Choose Model", ["Model 1 - Reusable vs Non-reusable", "Model 2 - Clean Reusable Only"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Choose model based on selection
    model = model1 if "vs Non" in model_choice else model2

    # Predict
    image = Image.open(uploaded_image).convert("RGB")
    results = model.predict(image)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        cls_id = int(boxes.cls[0].item())
        conf = boxes.conf[0].item()
        label = model.names[cls_id]
        explanation = get_explanation(label, conf)
        st.markdown(f"### {explanation}")
    else:
        st.warning("⚠️ No reusable oil detected in the image.")
