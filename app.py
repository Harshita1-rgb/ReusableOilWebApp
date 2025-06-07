import streamlit as st
from PIL import Image
import gdown
import os
from ultralytics import YOLO

# Streamlit UI
st.set_page_config(page_title="Reusable Oil Detector", layout="centered")
st.title("🛢️ Reusable Oil Detector")
st.markdown("Upload an image and choose a model to check oil quality.")

# Model info
model_paths = {
    "Model 1 - Reusable vs Non-reusable": {
        "url": "https://drive.google.com/uc?id=1zLG_0gNfV59YxEyovad0aU79rTnYylby",
        "local": "reusable_vs_nonreusable.pt"
    },
    "Model 2 - Clean Reusable Only": {
        "url": "https://drive.google.com/uc?id=1110aAQtjWpViRoSOI2B9s6eS7uFCaz7M",
        "local": "clean_reusable_model.pt"
    }
}

# Select model
model_choice = st.selectbox("Choose Model", list(model_paths.keys()))
model_info = model_paths[model_choice]

# Download model if needed
if not os.path.exists(model_info["local"]):
    with st.spinner("📥 Downloading model..."):
        gdown.download(model_info["url"], model_info["local"], quiet=False)

# Load model
model = YOLO(model_info["local"])

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing..."):
        results = model.predict(img)

    boxes = results[0].boxes

    if boxes and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0]) * 100
            label = model.names[cls_id]

            st.success(f"🟩 {label.capitalize()} detected with {conf:.2f}% confidence")

            # Human explanation section
            st.markdown("### 🔎 Why this prediction?")
            if label == "reusable":
                st.markdown("""
                - ✅ Clear golden/yellow oil color  
                - ✅ Oil detected in clean utensil  
                - ✅ No visible burnt residue or froth  
                - ✅ Background not greasy or dark  
                """)
            elif label == "nonreusable":
                st.markdown("""
                - ⚠️ Dark or black oil appearance  
                - ⚠️ Greasy or burnt pan detected  
                - ⚠️ Reflective or contaminated background  
                - ⚠️ Irregular texture may suggest overuse  
                """)
    else:
        st.warning("⚠️ No oil detected in the image.")
