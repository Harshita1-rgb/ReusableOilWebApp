import streamlit as st 
import onnxruntime as ort
from PIL import Image
import numpy as np

st.set_page_config(page_title="Reusable Oil Detector", layout="centered")
st.title("🛢️ Reusable Oil Detector")
st.markdown("Upload an image and choose a model to check oil quality.")

# Upload and model selection
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("Choose Model", ["Model 1 - Reusable vs Non-reusable", "Model 2 - Clean Reusable Only"])

# Load ONNX model
@st.cache_resource
def load_model(path):
    return ort.InferenceSession(path)

# Image preprocessing
def preprocess_image(image):
    img = image.resize((640, 640)).convert("RGB")
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dim
    return img_array

# Main logic
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    image = Image.open(uploaded_file).convert("RGB")

    # Set model path and class labels
    if model_choice == "Model 1 - Reusable vs Non-reusable":
        model_path = "reusable_vs_nonreusable.onnx"
        labels = ["reusable", "nonreusable"]
    else:
        model_path = "clean_reusable_model.onnx"
        labels = ["reusable"]

    model = load_model(model_path)
    input_tensor = preprocess_image(image)
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: input_tensor})

    predictions = outputs[0][0]  # Shape: (num_boxes, 6)
    detected = False

    for pred in predictions:
        confidence = pred[4]
        if confidence > 0.3:
            class_id = int(pred[5])
            if class_id < len(labels):
                label = labels[class_id]
                conf_pct = confidence * 100
                st.success(f"🟩 {label.capitalize()} detected with {conf_pct:.2f}% confidence")

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
                detected = True
            break

    if not detected:
        st.warning("⚠️ No oil detected in the image.")
