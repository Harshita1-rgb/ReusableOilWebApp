import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np

# Title
st.title("🛢️ Reusable Oil Detector")
st.write("Upload an image and choose a model to check oil quality.")

# Load model with caching
@st.cache_resource
def load_model(path):
    return ort.InferenceSession(path)

# Preprocess image
def preprocess(img):
    img = img.resize((640, 640)).convert('RGB')
    img = np.array(img).transpose(2, 0, 1)  # HWC to CHW
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# Postprocess predictions
def postprocess(outputs, model_type):
    predictions = outputs[0]
    if predictions.shape[1] == 6:  # reusable vs nonreusable
        cls_map = {0: "reusable", 1: "nonreusable"}
    else:  # clean reusable only
        cls_map = {0: "reusable"}
    
    results = []
    for pred in predictions[0]:
        conf = pred[4]
        cls_id = int(pred[5])
        if conf > 0.2:
            results.append((cls_map.get(cls_id, "unknown"), float(conf)))
    return results

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Choose model
model_choice = st.radio("Choose Model", ["Model 1 - Reusable vs Non-reusable", "Model 2 - Clean Reusable Only"])

# Run inference
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    img_input = preprocess(image)
    
    if model_choice == "Model 1 - Reusable vs Non-reusable":
        model_path = "reusable_vs_nonreusable.onnx"
    else:
        model_path = "clean_reusable_model.onnx"
    
    model = load_model(model_path)
    outputs = model.run(None, {"images": img_input})
    results = postprocess(outputs, model_choice)

    if results:
        for label, conf in results:
            st.success(f"🟩 {label.capitalize()} detected with {conf*100:.2f}% confidence")
    else:
        st.warning("⚠️ No reusable oil detected in the image.")
