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

# ==================== GLOBAL SETTINGS ====================
torch.set_grad_enabled(False)
DEVICE = torch.device("cpu")

st.set_page_config(page_title="Industrial AI System", layout="wide")
st.title("🛠️ Industrial AI Vision System")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Task Selection")
mode = st.sidebar.selectbox(
    "Choose Task",
    ["Object Detection + Human/Animal Classification", "Offline Industrial OCR"]
)

st.sidebar.header("Detection Settings")
det_conf = st.sidebar.slider("Detection Confidence", 0.1, 0.95, 0.5, 0.05)

st.sidebar.header("OCR Settings")
ocr_conf = st.sidebar.slider("OCR Confidence", 0.3, 0.95, 0.7, 0.05)

# ==================== PATHS ====================
DETECTOR_PATH = "faster_rcnn_object_detector_FAST.pth"
CLASSIFIER_PATH = "animal_classifier.pth"

# ==================== IMAGE LOADERS ====================
def load_image_no_rotate(uploaded_file):
    uploaded_file.seek(0)
    img = Image.open(uploaded_file).copy()
    img.info.pop("exif", None)
    return img

def cv2_imread_no_rotate(uploaded_file):
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ==================== MODELS ====================
@st.cache_resource
def load_detector():
    if not os.path.exists(DETECTOR_PATH):
        st.error("Detector model missing")
        st.stop()

    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(DETECTOR_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_classifier():
    if not os.path.exists(CLASSIFIER_PATH):
        st.error("Classifier model missing")
        st.stop()

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)

# ==================== TRANSFORMS ====================
classify_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CLASS_NAMES = {0: "Animal", 1: "Human"}

# ==================== PART A ====================
if mode == "Object Detection + Human/Animal Classification":

    detector = load_detector()
    classifier = load_classifier()

    def process_frame(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transforms.ToTensor()(rgb).unsqueeze(0)

        preds = detector(tensor)[0]
        keep = preds["scores"] > det_conf

        boxes = preds["boxes"][keep].cpu().numpy()
        scores = preds["scores"][keep].cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            inp = classify_tf(Image.fromarray(crop)).unsqueeze(0)
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

    uploaded = st.file_uploader("Upload Image or Video", ["jpg", "png", "mp4"])

    if uploaded:
        if uploaded.type.startswith("image"):
            img = cv2_imread_no_rotate(uploaded)
            out = process_frame(img.copy())
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

        else:
            st.video(uploaded)

# ==================== PART B ====================
else:
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

    uploaded = st.file_uploader("Upload Image", ["jpg", "png"])

    if uploaded:
        img = cv2_imread_no_rotate(uploaded)
        pre = ocr_preprocess(img)

        res = ocr.ocr(pre, cls=True)

        for line in res or []:
            for word in line:
                txt, conf = word[1]
                if conf > ocr_conf:
                    box = np.array(word[0], np.int32)
                    cv2.polylines(img, [box], True, (0, 255, 0), 2)
                    cv2.putText(img, txt, (box[0][0], box[0][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

st.caption("Offline • CPU-only • Industrial-grade")
