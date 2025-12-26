import streamlit as st
import torch
import cv2
import numpy as np
from torchvision.models import resnet18
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import tempfile
import os

# App Title and Description
st.title("Industrial Human & Animal Detection System")
st.markdown("""
This app detects objects (humans/animals) in images/videos using a custom-trained Faster R-CNN model, 
crops detections, and classifies them as Human or Animal with a 98.91% accurate ResNet18 classifier.
Upload an image or video below. Adjust detection threshold in the sidebar.
""")

# Sidebar for Threshold
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
st.sidebar.markdown("Higher threshold = fewer but more confident detections.")

# Model Paths (adjust if needed)
DETECTOR_PATH = "faster_rcnn_object_detector_FAST.pth"
CLASSIFIER_PATH = "animal_classifier.pth"

# Load Detector
@st.cache_resource
def load_detector():
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, num_classes=2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(DETECTOR_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load Classifier
@st.cache_resource
def load_classifier():
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 0: Animal, 1: Human
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

detector = load_detector()
classifier = load_classifier()

# Transform for Classifier
classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to Process Image
def process_image(image, threshold):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)
    
    with torch.no_grad():
        predictions = detector(img_tensor)[0]
    
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    output_img = image.copy()
    crops = []
    class_results = []
    
    for box, score in zip(boxes[scores > threshold], scores[scores > threshold]):
        x1, y1, x2, y2 = map(int, box)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        # Draw detection box
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(output_img, f"Object ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Classify
        crop_pil = Image.fromarray(crop)
        input_tensor = classify_transform(crop_pil).unsqueeze(0)
        with torch.no_grad():
            output = classifier(input_tensor)
        pred_idx = output.argmax(1).item()
        label = "Animal" if pred_idx == 0 else "Human"
        conf = torch.softmax(output, dim=1).max().item()
        
        crops.append(crop)
        class_results.append((label, conf))
        
        # Draw classification on output (optional, but focused UI)
        color = (0, 0, 255) if pred_idx == 0 else (0, 255, 0)
        cv2.putText(output_img, f"{label} ({conf:.2f})", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return output_img, crops, class_results

# Function to Process Video
def process_video(video_path, threshold):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4").name
    writer = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, _, _ = process_image(frame, threshold)
        writer.write(processed_frame)
        frame_count += 1
        if frame_count % 10 == 0:
            st.write(f"Processing frame {frame_count}...")
    
    cap.release()
    writer.release()
    return temp_output

# Upload Section
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == "image":
        st.subheader("Uploaded Image")
        image = np.array(Image.open(uploaded_file))
        st.image(image, channels="BGR", use_column_width=True)
        
        st.subheader("Detection Results")
        detected_img, crops, results = process_image(image, confidence_threshold)
        st.image(detected_img, channels="BGR", caption="Detected Objects", use_column_width=True)
        
        if crops:
            st.subheader("Cropped Detections and Classifications")
            cols = st.columns(min(len(crops), 3))  # Display up to 3 per row
            for i, (crop, (label, conf)) in enumerate(zip(crops, results)):
                with cols[i % 3]:
                    st.image(crop, channels="RGB", caption=f"Crop {i+1}")
                    st.write(f"Class: {label} (Conf: {conf:.2f})")
    
    elif file_type == "video":
        st.subheader("Uploaded Video")
        temp_input = tempfile.NamedTemporaryFile(delete=False).name
        with open(temp_input, "wb") as f:
            f.write(uploaded_file.read())
        st.video(temp_input)
        
        st.subheader("Processing Video...")
        output_video_path = process_video(temp_input, confidence_threshold)
        
        st.subheader("Processed Video (Annotated with Classes)")
        st.video(output_video_path)
        
        # Auto Download
        with open(output_video_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
        
        os.unlink(temp_input)  # Clean up

else:
    st.info("Upload an image or video to start.")