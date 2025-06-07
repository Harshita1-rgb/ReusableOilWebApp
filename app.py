import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np

st.set_page_config(page_title="Reusable Oil Detector", layout="centered")

st.title("🛢️ Reusable Oil Detector")
st.markdown("Upload an image and choose a model to check oil qualiimport streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

# Title
st.title("🛢️ Reusable Oil Detector")
st.write("Upload an image and choose a model to check oil quality.")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Model Selector
model_choice = st.selectbox("Choose Model", ["Model 1 - Reusable vs Non-reusable", "Model 2 - Clean Reusable Only"])

# Load ONNX model
@st.cache_resource
def load_model(model_path):
    return ort.InferenceSession(model_path)

def preprocess_image(image):
    img = image.resize((640, 640)).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)[None, ...]
    return img_np

def postprocess_output(output, model_type):
    scores = output[0][:, 4]  # Objectness scores
    classes = output[0][:, 5].astype(int)  # Class predictions
    best_idx = np.argmax(scores)
    confidence = scores[best_idx]
    label_id = classes[best_idx]

    if model_type == "Model 1 - Reusable vs Non-reusable":
        label_map = {0: "Reusable", 1: "Non-reusable"}
    else:
        label_map = {0: "Reusable"}

    label = label_map.get(label_id, "Unknown")
    explanation = explain_reasoning(label)
    return label, confidence, explanation

def explain_reasoning(label):
    if label == "Reusable":
        return "✅ The oil appears clear with minimal food residue or discoloration, indicating it is likely reusable."
    elif label == "Non-reusable":
        return "⚠️ The oil seems dark, cloudy, or contains burnt particles, suggesting it is no longer safe for reuse."
    else:
        return "⚠️ The model could not confidently determine the oil condition."

# Run Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model_choice == "Model 1 - Reusable vs Non-reusable":
        model_path = "reusable_vs_nonreusable.onnx"
    else:
        model_path = "clean_reusable_model.onnx"

    session = load_model(model_path)
    input_tensor = preprocess_image(image)
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    
    if len(outputs) > 0 and outputs[0].shape[1] > 0:
        label, confidence, explanation = postprocess_output(outputs, model_choice)
        st.success(f"{label} detected with {confidence * 100:.2f}% confidence")
        st.info(explanation)
    else:
        st.warning("⚠️ No oil detected. Please upload a clearer image.")
ty.")

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
