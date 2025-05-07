import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import defaultdict
import gdown
import requests

# Google Drive file ID
file_id = "1mT6KhX38bV5km_VP81SxRAhkscf4E4uy"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Send GET request to download the file
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    with open("best.pt", "wb") as f:
        f.write(response.content)
    print("best.pt downloaded successfully!")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")


# Page config
st.set_page_config(
    page_title="NIT Warangal | Steel Surface Defect Detection",
    page_icon="🛠️",
    layout="wide"
)

# Minimal styling to avoid scroll but keep default background
st.markdown("""
    <style>
    .reportview-container .main {
        overflow: hidden;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    h1, h2 {
        color: #0a3d62;
        text-align: center;
    }
    .stButton>button {
        background-color: #0a3d62;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
    .stFileUploader label {
        color: #0a3d62;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("National Institute of Technology, Warangal")
st.subheader("🛠️ AI-Based Steel Surface Defect Detection System")
st.markdown("Upload an image of a **hot rolled steel strip** to detect and classify surface defects using a **YOLOv8 deep learning model**.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

# Defect Knowledge Base
defect_knowledge = {
    "Crazing": {
        "Cause": "Tensile stress beyond material limit due to cooling issues or high rolling speed.",
        "Prevention": "Optimize rolling speed and ensure uniform cooling."
    },
    "Patches": {
        "Cause": "Local oxidation or improper cleaning before rolling.",
        "Prevention": "Maintain surface cleanliness and control mill scale formation."
    },
    "Pitted_surface": {
        "Cause": "Localized corrosion or trapped air bubbles during rolling.",
        "Prevention": "Improve descaling processes and surface inspection."
    },
    "Rolled-in_scale": {
        "Cause": "Oxide scales not removed properly before rolling.",
        "Prevention": "Enhance descaling efficiency and pre-cleaning."
    },
    "Scratches": {
        "Cause": "Abrasive particles or improper handling.",
        "Prevention": "Maintain clean rollers and handling equipment."
    },
    "Inclusion": {
        "Cause": "Non-metallic particles embedded during manufacturing.",
        "Prevention": "Use high-quality raw materials and controlled processing."
    }
}

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Run detection
        results = model(image_np)
        annotated_img = results[0].plot()

        # Collect defect info
        grouped_defects = defaultdict(list)
        for box in results[0].boxes.data.tolist():
            _, _, _, _, score, cls_id = box
            class_name = results[0].names[int(cls_id)]
            grouped_defects[class_name].append(score)

        # Display side by side
        col1, col2 = st.columns([1, 1.3])

        with col1:
            st.image(annotated_img, caption="Detected Defects", use_column_width=True)

        with col2:
            st.subheader("📋 Detected Defects & Technical Details")
            for defect, scores in grouped_defects.items():
                class_key = defect.strip().replace(" ", "_").capitalize()
                st.markdown(f"### 🔹 {defect}")
                for idx, conf in enumerate(scores):
                    st.markdown(f"- Confidence {idx+1}: **{conf:.2f}**")

                if class_key in defect_knowledge:
                    st.markdown(f"**🛠 Cause:** {defect_knowledge[class_key]['Cause']}")
                    st.markdown(f"**✅ Prevention:** {defect_knowledge[class_key]['Prevention']}")
                else:
                    st.warning("⚠️ No information found for this defect.")

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload a valid image file (jpg, jpeg, png, bmp, tiff, webp).")
