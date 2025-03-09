import streamlit as st
import ultralytics
from ultralytics import YOLO
from PIL import Image
import time
import numpy as np
import cv2
from streamlit_image_select import image_select

# Streamlit UI
st.title("PIDetect - P&ID Tool")
st.write("Upload or Select a P&ID Image")

# Load YOLO model
model2 = YOLO("finaltrain_best.pt")

# Class name mapping
class_name_mapping = {
    "0": "Ball Valve 1",
    "1": "Ball Valve 2",
    "2": "Ball Valve 3",
    "3": "Onsheet Connector",
    "4": "Centrifugal Fan",
    "5": "IHTL",
    "6": "Pneumatic Signal",
    "7": "NP"
}

# Color mapping for bounding boxes
class_color_mapping = {
    "0": (255, 0, 0),
    "1": (0, 255, 0),
    "2": (0, 0, 255),
    "3": (255, 255, 0),
    "4": (255, 0, 255),
    "5": (0, 255, 255),
    "6": (128, 0, 128),
    "7": (0, 128, 128)
}

# Sidebar for class filtering
st.sidebar.title("Filter Classes")
selected_classes = {}
for class_id, class_name in class_name_mapping.items():
    selected_classes[class_id] = st.sidebar.checkbox(class_name, value=True)

# Image Selector
img_path = image_select("Select P&ID Image", ["assets/PIDs/2.jpeg", "assets/PIDs/3.jpeg", "assets/PIDs/5.png", "assets/PIDs/6.png"])

# File Uploader
uploaded_file = st.file_uploader("Or Upload an Image", type=["png", "jpg", "jpeg"])

# Determine which image to process
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif img_path:
    image = Image.open(img_path)

# Process the selected image
if image is not None:
    st.image(image, caption="Selected Image", use_column_width=True)

    if st.button("Predict"):
        # Convert image to numpy array for YOLO processing
        image_np = np.array(image)

        # Run inference
        results = model2(image_np, conf=0.1)
        with st.status("Processing P&ID..."):
            st.write("Analyzing Data...")
            time.sleep(1)
            st.write("Initializing Trained Computer Vision Model")
            time.sleep(2)
            st.write("CV Model Processing...")
            time.sleep(1)

        # Draw bounding boxes
        for box in results[0].boxes:
            original_class = results[0].names[int(box.cls)]
            if selected_classes.get(str(int(box.cls)), False):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                label = class_name_mapping.get(original_class, original_class)
                color = class_color_mapping.get(str(int(box.cls)), (0, 255, 0))

                cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_np, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Convert back to RGB for Streamlit display
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Processed Image", use_column_width=True)
