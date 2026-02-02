import os
import sys
import h5py
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# --- 1. CONFIGURATION ---
MODEL_PATH = "diabetic_retinopathy_mobilenet.h5"
LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}

# --- 2. INTELLIGENT MODEL BUILDER (DO NOT TOUCH THIS SECTION) ---
def get_hidden_layer_units(filepath):
    print(f"🕵️  Scanning {filepath} to detect architecture...")
    try:
        with h5py.File(filepath, 'r') as f:
            def find_dense_shape(name, obj):
                if isinstance(obj, h5py.Dataset) and 'kernel' in name:
                    shape = obj.shape
                    if shape[0] == 1280 and shape[1] != 5:
                        return shape[1]
            
            hidden_units = None
            if 'model_weights' in f:
                weight_groups = f['model_weights']
                for layer_name in weight_groups:
                    layer_group = weight_groups[layer_name]
                    if hasattr(layer_group, 'visititems'):
                        res = []
                        def visitor(name, obj):
                            val = find_dense_shape(name, obj)
                            if val: res.append(val)
                        layer_group.visititems(visitor)
                        if res:
                            hidden_units = res[0]
                            break
            return hidden_units
    except Exception as e:
        return None

def build_adaptive_model(hidden_units=None):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False, 
        weights=None
    )
    base_model.trainable = False 

    layers = [base_model, tf.keras.layers.GlobalAveragePooling2D()]
    if hidden_units:
        layers.append(tf.keras.layers.Dense(hidden_units, activation='relu'))
    layers.append(tf.keras.layers.Dense(5, activation='softmax'))
    
    return tf.keras.Sequential(layers)

# --- 3. LOAD PROCESS ---
detected_units = get_hidden_layer_units(MODEL_PATH)
model = build_adaptive_model(detected_units)

try:
    model.build((None, 224, 224, 3))
    model.load_weights(MODEL_PATH)
    print("✅ SUCCESS: Model loaded and ready!")
except Exception as e:
    print(f"❌ Load failed: {e}")
    try:
        model = build_adaptive_model(1024)
        model.build((None, 224, 224, 3))
        model.load_weights(MODEL_PATH)
        print("✅ SUCCESS: Model loaded (Fallback)!")
    except:
        model = None

# --- 4. PREDICTION FUNCTION ---
def predict_retinopathy(image):
    if model is None:
        return None
    if image is None:
        return None

    try:
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]
        return {LABELS[i]: float(prediction[i]) for i in range(len(LABELS))}
    except Exception:
        return None

# --- 5. UI DESIGN ---

custom_css = """
.container { max-width: 1200px; margin: auto; padding-top: 20px; }
#header { text-align: center; margin-bottom: 30px; }
#header h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #059669; font-size: 2.5rem; }
.upload-box { border-radius: 15px !important; border: 2px dashed #10b981 !important; }
.result-box { border-radius: 15px !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important; }
.analyze-btn { background: linear-gradient(90deg, #059669 0%, #10b981 100%) !important; border: none !important; color: white !important; font-weight: bold !important; transition: transform 0.1s; }
.analyze-btn:hover { transform: scale(1.02); }
button[aria-label="Fullscreen"] { display: none !important; }
"""


theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
).set(
    body_background_fill="#f0fdf4",
    block_background_fill="#ffffff",
    block_border_width="0px",
    shadow_drop_lg="0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)"
)

with gr.Blocks(theme=theme, css=custom_css, title="RetinAI") as demo:
    
    # Header Section
    with gr.Column(elem_id="header"):
        gr.Markdown(
            """
            # RetinAI Diagnostic System
            """
        )
    
    # Main Interface
    with gr.Row():
        
        # LEFT COLUMN: Input
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Fundus Scan")
            
            input_image = gr.Image(
                type="pil", 
                label="Drop Image Here", 
                sources=["upload"],  
                height=350,
                elem_classes="upload-box"
            )
            
            gr.Markdown(
                """
                <div style="color: #64748b; font-size: 0.9em; margin-top: 10px;">
                <b>Supported Formats:</b> JPG, PNG, JPEG <br>
                <b>Note:</b> Ensure the retina is centered and fully visible.
                </div>
                """
            )
            
            analyze_btn = gr.Button("Analyze Retina", elem_classes="analyze-btn", size="lg")

        # RIGHT COLUMN: Output
        with gr.Column(scale=1):
            gr.Markdown("### 2. AI Prediction Analysis")
            
            output_label = gr.Label(
                num_top_classes=3, 
                label="Confidence Score",
                elem_classes="result-box"
            )
            
            # Disclaimer Accordion
            with gr.Accordion("⚠️ Medical Disclaimer & Privacy", open=False):
                gr.Markdown(
                    """
                    * **Not a Diagnosis:** This tool is for educational and research demonstration purposes only.
                    * **Consult a Professional:** Always verify results with a certified ophthalmologist.
                    * **Privacy:** Images processed here are not saved to any server permanently.
                    """
                )

    # Footer
    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align: center; color: #94a3b8; margin-top: 40px;">
            Developed with MobileNetV2 & TensorFlow | © 2026 Archa Vivek | MIT Licensed
            </div>
            """
        )

    # Logic
    analyze_btn.click(predict_retinopathy, inputs=input_image, outputs=output_label)

if __name__ == "__main__":
    demo.launch(share=True)