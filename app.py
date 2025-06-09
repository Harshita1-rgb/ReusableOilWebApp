import gradio as gr
from ultralytics import YOLO
import gdown
import os

# Define model links and local paths
model_paths = {
    "Model 1 - Reusable vs Nonreusable": {
        "url": "https://drive.google.com/uc?id=1zLG_0gNfV59YxEyovad0aU79rTnYylby",
        "local": "model1.pt"
    },
    "Model 2 - Clean Reusable Only": {
        "url": "https://drive.google.com/uc?id=1110aAQtjWpViRoSOI2B9s6eS7uFCaz7M",
        "local": "model2.pt"
    }
}

# Load selected model (downloads if not present)
def load_model(model_choice):
    model_info = model_paths[model_choice]
    if not os.path.exists(model_info["local"]):
        gdown.download(model_info["url"], model_info["local"], quiet=False)
    return YOLO(model_info["local"])

# Prediction function
def predict(image, model_choice):
    model = load_model(model_choice)
    results = model.predict(image, imgsz=640, conf=0.5)
    names = model.names
    label_id = int(results[0].boxes.cls[0])
    label = names[label_id]
    confidence = float(results[0].boxes.conf[0]) * 100

    explanation = ""
    if label == "reusable":
        explanation = "- ✅ Clear golden/yellow oil\n- ✅ Clean utensil\n- ✅ No visible burnt residue or froth"
    elif label == "nonreusable":
        explanation = "- ⚠️ Dark or black oil\n- ⚠️ Greasy/burnt pan\n- ⚠️ Contaminated background or froth"

    return f"**Prediction:** {label.upper()}\n**Confidence:** {confidence:.2f}%\n\n**Why?**\n{explanation}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🛢️ Cooking Oil Quality Checker (AI-Powered)")
    model_choice = gr.Dropdown(choices=list(model_paths.keys()), label="Choose Model")
    image_input = gr.Image(type="pil")
    output = gr.Markdown()
    predict_btn = gr.Button("Predict")
    predict_btn.click(fn=predict, inputs=[image_input, model_choice], outputs=output)

demo.launch()
