import streamlit as st
from PIL import Image
import numpy as np
import torch
import os
import cv2

MODEL_PATH = 'yolov5s.pt'
if not os.path.exists(MODEL_PATH):
    st.error('Model file not found. Please ensure yolov5s.pt is in the working directory.')
    st.stop()

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

model = load_model(MODEL_PATH)

def draw_boxes(image, detections):
    for _, row in detections.iterrows():
        start_point = (int(row['xmin']), int(row['ymin']))
        end_point = (int(row['xmax']), int(row['ymax']))
        color = (0, 255, 0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        label = row['name']
        image = cv2.putText(image, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image

st.title('Object Detection with YOLO')
st.write('Upload an image and press "Analyze" to detect objects.')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('')
    if st.button('Analyze'):
        st.write('Analyzing the image, please wait...')
        image_np = np.array(image)
        results = model(image_np)
        detections = results.pandas().xyxy[0]
        image_with_boxes = draw_boxes(image_np.copy(), detections)
        num_objects = len(detections)
        st.write(f'Number of detected objects: {num_objects}')
        detected_objects = detections['name'].unique()
        st.write('Detected objects:')
        for obj in detected_objects:
            st.write(obj)
        st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)
