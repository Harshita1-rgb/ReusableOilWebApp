import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np

st.set_page_config(page_title="Reusable Oil Detector", layout="centered")

st.title("🛢️ Reusable Oil Detector")
st.markdown("Upload an image and choose a model to check oil quality.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("Choose Model", [
    "Model 1 - Reusable vs Non-reusable",
    "Model 2 - Clean Reusable Only"
])

# Load ONNX model
@st.cache_resource
def load_model(path):
    return ort.InferenceSession(path)

if model_choice == "Model 1 - Reusable vs Non-reusable":
    model_path = "reusable_vs_nonreusable.onnx"
    labels = ["reusable", "nonreusable"]
else:
    model_path = "clean_reusable_model.onnx"
    labels = ["reusable"]

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img = Image.open(uploaded_file).convert("RGB")
    model = load_model(model_path)

    # Preprocess image
    img_resized = img.resize((640, 640))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Run inference
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: img_array})

    predictions = outputs[0][0]

    detected = False
    for pred in predictions:
        confidence = pred[4]
        if confidence > 0.3:  # Confidence threshold
            class_id = int(pred[5])
            label = labels[class_id] if class_id < len(labels) else "unknown"
            st.success(f"✅ {label.capitalize()} detected with {confidence * 100:.2f}% confidence")
            detected = True
            break

    if not detected:
        st.warning("⚠️ No reusable oil detected in the image.")
