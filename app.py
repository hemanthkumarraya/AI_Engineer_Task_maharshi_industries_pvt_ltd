import streamlit as st
import torch
import cv2
import numpy as np
from torchvision.models import resnet18
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
from paddleocr import PaddleOCR
import tempfile
import os
import json

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Industrial AI System", layout="wide")
st.title("🛠️ Industrial AI Vision System")
st.markdown("**Fully Offline Computer Vision Pipeline**")

torch.set_grad_enabled(False)
DEVICE = torch.device("cpu")

# ==================== NO AUTO-ROTATION HELPERS ====================
def load_image_no_rotate(uploaded_file):
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    img = img.copy()
    img.info.pop("exif", None)  # REMOVE EXIF → NO ROTATION
    return img

def cv2_imread_no_rotate(uploaded_file):
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ==================== SIDEBAR ====================
mode = st.sidebar.selectbox(
    "Choose Task",
    ["Object Detection + Human/Animal Classification", "Offline Industrial OCR"]
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 0.95, 0.5, 0.05
)

# ==================== MODEL PATHS ====================
DETECTOR_PATH = "faster_rcnn_object_detector_FAST.pth"
CLASSIFIER_PATH = "animal_classifier.pth"

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_detector():
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(DETECTOR_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_classifier():
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)

# ==================== TRANSFORMS ====================
classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CLASS_NAMES = {0: "Animal", 1: "Human"}

# ==================== PART A ====================
if mode == "Object Detection + Human/Animal Classification":
    st.header("🔍 Detection + Classification")

    detector = load_detector()
    classifier = load_classifier()

    def process_frame(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transforms.ToTensor()(rgb).unsqueeze(0)

        preds = detector(tensor)[0]
        keep = preds["scores"] > conf_threshold

        boxes = preds["boxes"][keep].cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            inp = classify_transform(Image.fromarray(crop)).unsqueeze(0)
            out = classifier(inp)
            idx = out.argmax(1).item()
            conf = torch.softmax(out, 1)[0][idx].item()
            label = CLASS_NAMES[idx]

            color = (0, 255, 0) if label == "Human" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame

    uploaded = st.file_uploader("Upload Image or Video", ["jpg", "jpeg", "png", "mp4"])

    if uploaded:
        if uploaded.type.startswith("image"):
            img_pil = load_image_no_rotate(uploaded)
            img_cv = cv2_imread_no_rotate(uploaded)

            st.subheader("Original (No Rotation)")
            st.image(img_pil, use_column_width=True)

            out = process_frame(img_cv.copy())
            st.subheader("Result")
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

        else:
            st.video(uploaded)

# ==================== PART B ====================
else:
    st.header("📄 Offline Industrial OCR")
    ocr = load_ocr()

    def ocr_preprocess(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(4.0, (8, 8)).apply(gray)
        den = cv2.fastNlMeansDenoising(clahe, h=15)
        bin = cv2.adaptiveThreshold(
            den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
        return cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)

    uploaded = st.file_uploader("Upload Image", ["jpg", "jpeg", "png"])

    if uploaded:
        img_pil = load_image_no_rotate(uploaded)
        img_cv = cv2_imread_no_rotate(uploaded)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original (No Rotation)")
            st.image(img_pil, use_column_width=True)

        pre = ocr_preprocess(img_cv)
        with col2:
            st.subheader("Preprocessed")
            st.image(cv2.cvtColor(pre, cv2.COLOR_BGR2RGB), use_column_width=True)

        res = ocr.ocr(pre, cls=True)

        for line in res or []:
            for word in line:
                text, conf = word[1]
                if conf > conf_threshold:
                    box = np.array(word[0], np.int32)
                    cv2.polylines(img_cv, [box], True, (0, 255, 0), 2)
                    cv2.putText(img_cv, text,
                                (box[0][0], box[0][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)

        st.subheader("OCR Result")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_column_width=True)

st.caption("Fully offline • CPU-only • No EXIF rotation • Industrial-grade")
