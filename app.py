import streamlit as st
import torch
import cv2
import numpy as np
from torchvision.models import resnet18
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ExifTags
import tempfile
import os

# Fix image orientation from EXIF (prevents unwanted rotation)
def fix_image_orientation(img_pil):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img_pil._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                img_pil = img_pil.rotate(180, expand=True)
            elif orientation == 6:
                img_pil = img_pil.rotate(270, expand=True)
            elif orientation == 8:
                img_pil = img_pil.rotate(90, expand=True)
    except (AttributeError, KeyError, TypeError):
        # No EXIF or no orientation tag → do nothing
        pass
    return img_pil

# App Title and Description
st.set_page_config(page_title="Human & Animal Detection", layout="wide")
st.title("Industrial Human & Animal Detection System")
st.markdown("""
This app uses a **custom-trained two-stage model**:
1. Faster R-CNN (MobileNetV3) detects living objects
2. ResNet18 classifier (98.91% accuracy) labels them as **Human** or **Animal**

Upload an image or video. Adjust confidence threshold in the sidebar.
""")

# Sidebar
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.1, max_value=0.95, value=0.5, step=0.05,
    help="Higher = stricter (fewer false detections)"
)

# Model Paths
DETECTOR_PATH = "faster_rcnn_object_detector_FAST.pth"
CLASSIFIER_PATH = "animal_classifier.pth"

# Load Models (cached + CPU only for Streamlit Cloud)
@st.cache_resource
def load_detector():
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, num_classes=2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(DETECTOR_PATH, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_classifier():
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location="cpu"))
    model.eval()
    return model

detector = load_detector()
classifier = load_classifier()

# Classifier transform
classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Process single image
def process_image(image_bgr, threshold):
    # Convert to RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = fix_image_orientation(img_pil)  # Fix rotation
    img_rgb = np.array(img_pil)

    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)

    with torch.no_grad():
        predictions = detector(img_tensor)[0]

    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    output_img = image_bgr.copy()
    crops = []
    class_results = []

    for box, score in zip(boxes[scores > threshold], scores[scores > threshold]):
        x1, y1, x2, y2 = map(int, box)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Detection box (blue)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 144, 30), 3)
        cv2.putText(output_img, f"Detect: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 144, 30), 2)

        # Classification
        crop_pil = Image.fromarray(crop)
        input_tensor = classify_transform(crop_pil).unsqueeze(0)
        with torch.no_grad():
            output = classifier(input_tensor)
        pred_idx = output.argmax(1).item()
        label = "Animal" if pred_idx == 0 else "Human"
        conf = torch.softmax(output, dim=1)[0][pred_idx].item()

        crops.append(crop)
        class_results.append((label, conf))

        # Final label on image
        color = (0, 0, 255) if label == "Animal" else (0, 255, 0)  # Red=Animal, Green=Human
        cv2.putText(output_img, f"{label} ({conf:.2f})", (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return output_img, crops, class_results

# Process video
def process_video(video_path, threshold):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, _, _ = process_image(frame, threshold)
        writer.write(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    cap.release()
    writer.release()
    progress_bar.empty()
    return temp_output

# File Upload
uploaded_file = st.file_uploader("Upload Image (jpg/png) or Video (mp4)", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        # Image Processing
        st.subheader("Original Image")
        image_bytes = uploaded_file.read()
        image_pil = Image.open(uploaded_file)
        image_pil = fix_image_orientation(image_pil)  # Fix before display
        st.image(image_pil, use_column_width=True)

        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting and classifying..."):
            detected_img, crops, results = process_image(image_cv, confidence_threshold)

        st.subheader("Detection & Classification Result")
        st.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), use_column_width=True)

        if crops:
            st.subheader("Individual Crops & Classifications")
            cols = st.columns(min(len(crops), 4))
            for i, (crop, (label, conf)) in enumerate(zip(crops, results)):
                with cols[i % 4]:
                    st.image(crop, caption=f"{label} ({conf:.2f})", use_column_width=True)

    elif "video" in file_type:
        # Video Processing
        st.subheader("Original Video")
        st.video(uploaded_file)

        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_input, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.subheader("Processing Video...")
        with st.spinner("This may take a few minutes depending on video length..."):
            output_path = process_video(temp_input, confidence_threshold)

        st.subheader("Processed Video (with Human/Animal Labels)")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Download Annotated Video",
                data=f,
                file_name="human_animal_detection_output.mp4",
                mime="video/mp4"
            )

        # Cleanup
        os.unlink(temp_input)
        os.unlink(output_path)

else:
    st.info("👆 Please upload an image or video to begin detection.")
    st.markdown("### Supported: JPG, PNG, MP4")
