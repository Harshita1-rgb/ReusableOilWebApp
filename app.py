import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np

st.set_page_config(page_title="Reusable Oil Detector", layout="centered")
st.title("🛢️ Reusable Oil Detector")
st.markdown("Upload an image and choose a model to check oil quality.")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Model Selector
model_choice = st.selectbox("Choose Model", ["Model 1 - Reusable vs Non-reusable", "Model 2 - Clean Reusable Only"])

# Load ONNX Model
@st.cache_resource
def load_model(path):
    return ort.InferenceSession(path)

# Preprocess Image
def preprocess_image(image):
    img = image.resize((640, 640)).convert("RGB")
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    return img_array

# Reason Explanation
def explain(label):
    if label == "Reusable":
        return "🟢 The oil is clear and light in color, with minimal particles or discoloration — looks reusable."
    elif label == "Non-reusable":
        return "🔴 The oil appears dark, cloudy, or contains visible burnt particles — not safe for reuse."
    else:
        return ""

# Prediction Logic
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    image = Image.open(uploaded_file).convert("RGB")

    if model_choice == "Model 1 - Reusable vs Non-reusable":
        model_path = "reusable_vs_nonreusable.onnx"
        labels = ["Reusable", "Non-reusable"]
    else:
        model_path = "clean_reusable_model.onnx"
        labels = ["Reusable"]

    model = load_model(model_path)
    input_tensor = preprocess_image(image)

    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: input_tensor})

    predictions = outputs[0][0]
    detected = False

    for pred in predictions:
        confidence = pred[4]
        if confidence > 0.3:  # Confidence threshold
            class_id = int(pred[5])
            label = labels[class_id] if class_id < len(labels) else "Unknown"
            st.success(f"✅ {label} detected with {confidence * 100:.2f}% confidence")
            st.info(explain(label))
            detected = True
            break

    if not detected:
        st.warning("⚠️ No clear detection. The image might be blurry, dark, or doesn't match training examples.")
