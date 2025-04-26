import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv8 Face Detection", layout="centered")
st.title("YOLOv8 Human Face Detection (Image / Video / Webcam)")

# Load YOLOv8 model
model = YOLO("yolov8n-face.pt")  # Adjust path if needed

# Selection box for input type
option = st.selectbox("Choose input type:", ["Select", "Image", "Video", "Webcam"])

# IMAGE processing
if option == "Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        file_type = file.type
        if file_type.startswith("image"):
            image = Image.open(file).convert("RGB")
            image_np = np.array(image)

            results = model(image_np, conf=0.4)
            annotated_image = results[0].plot()

            st.image(annotated_image, caption="Detected Faces", use_column_width=True)

            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                st.write(f"Class: {model.names[cls_id]} | Confidence: {conf:.2f}")

# VIDEO processing
elif option == "Video":
    file = st.file_uploader("Upload a video", type=["mp4"])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame, conf=0.4)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="RGB", caption=f"Frame {frame_count}", use_column_width=True)

        cap.release()

# WEBCAM processing
elif option == "Webcam":
    if st.button("Start Webcam Face Detection"):
        st.warning("To stop webcam")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, conf=0.4)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="RGB", caption=f"Webcam Frame {frame_count}", use_column_width=True)

        cap.release()
