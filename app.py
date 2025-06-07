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
model_choice = st.selectbox("Choose Model", [
    "Model 1 - Reusable vs Non-reusable",
    "Model 2 - Clean Reusable Only"
])

# Load ONNX model
@st.cache_resource
def load_model(path):
    return ort.InferenceSession(path)

# Preprocess image
def preprocess(image):
    image = image.resize((640, 640)).convert("RGB")
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)  # HWC to CHW
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension
    return img_np

# Explain reasoning
def explain(label):
    if label == "Reusable":
        return "✅ The oil appears clear with minimal food residue or discoloration, indicating it is likely reusable."
    elif label == "Non-reusable":
        return "⚠️ The oil seems dark, cloudy, or contains burnt particles, suggesting it is no longer safe for reuse."
    else:
        return "⚠️ The model could not confidently determine the oil condition."

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Choose model path and labels
    if model_choice == "Model 1 - Reusable vs Non-reusable":
        model_path = "reusable_vs_nonreusable.onnx"
        labels = ["Reusable", "Non-reusable"]
    else:
        model_path = "clean_reusable_model.onnx"
        labels = ["Reusable"]

    model = load_model(model_path)
    input_tensor = preprocess(image)
    input_name = model.get_inputs()[0].name

    outputs = model.run(None, {input_name: input_tensor})
    predictions = outputs[0][0]

    detected = False
    for pred in predictions:
        confidence = pred[4]
        if confidence > 0.3:
            class_id = int(pred[5])
            label = labels[class_id] if class_id < len(labels) else "Unknown"
            st.success(f"{label} detected with {confidence * 100:.2f}% confidence")
            st.info(explain(label))
            detected = True
            break

    if not detected:
        st.warning("⚠️ No reusable oil detected in the image.")
