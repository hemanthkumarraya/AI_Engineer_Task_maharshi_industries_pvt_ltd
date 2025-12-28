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

# Page Config
st.set_page_config(page_title="Industrial AI System", layout="wide")
st.title("🛠️ Industrial AI Vision System")
st.markdown("**Offline Computer Vision Pipeline** – Detection, Classification & OCR")

# Sidebar: Mode Selection
mode = st.sidebar.selectbox(
    "Choose Task",
    ["Object Detection + Human/Animal Classification", "Offline Industrial OCR"]
)

# Common Settings
st.sidebar.header("⚙️ Common Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.95, 0.5, 0.05)

# Model Paths
DETECTOR_PATH = "faster_rcnn_object_detector_FAST.pth"
CLASSIFIER_PATH = "animal_classifier.pth"

# Load Models (cached)
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

@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

# Transforms
classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================ PART A: Detection + Classification ================
if mode == "Object Detection + Human/Animal Classification":
    st.header("🔍 Object Detection + Human/Animal Classification")
    st.markdown("Detects living objects → Classifies as **Human** (Green) or **Animal** (Red)")

    detector = load_detector()
    classifier = load_classifier()

    def process_image_part_a(image_bgr, threshold):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)

        with torch.no_grad():
            predictions = detector(img_tensor)[0]

        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()

        output_img = image_bgr.copy()
        crops = []
        results = []

        for box, score in zip(boxes[scores > threshold], scores[scores > threshold]):
            x1, y1, x2, y2 = map(int, box)
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0: continue

            # Classification
            crop_pil = Image.fromarray(crop)
            input_tensor = classify_transform(crop_pil).unsqueeze(0)
            with torch.no_grad():
                output = classifier(input_tensor)
            pred_idx = output.argmax(1).item()
            label = "Animal" if pred_idx == 0 else "Human"
            conf = torch.softmax(output, dim=1)[0][pred_idx].item()

            crops.append(crop)
            results.append((label, conf))

            # Draw on output
            color = (0, 0, 255) if label == "Animal" else (0, 255, 0)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(output_img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        return output_img, crops, results

    uploaded = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4"], key="parta")

    if uploaded:
        if uploaded.type.startswith('image'):
            image = np.array(Image.open(uploaded))
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            result_img, crops, results = process_image_part_a(image_bgr, conf_threshold)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)

            if crops:
                st.subheader("Cropped Classifications")
                cols = st.columns(min(len(crops), 4))
                for i, (crop, (label, conf)) in enumerate(zip(crops, results)):
                    with cols[i % 4]:
                        st.image(crop, caption=f"{label} ({conf:.2f})")

        else:  # Video
            st.video(uploaded)
            if st.button("Process Video"):
                temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                with open(temp_in, "wb") as f:
                    f.write(uploaded.getbuffer())

                cap = cv2.VideoCapture(temp_in)
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                writer = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                progress = st.progress(0)
                frame_count = 0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    processed, _, _ = process_image_part_a(frame, conf_threshold)
                    writer.write(processed)
                    frame_count += 1
                    progress.progress(frame_count / total)

                cap.release()
                writer.release()
                progress.empty()

                st.video(temp_out)
                with open(temp_out, "rb") as f:
                    st.download_button("Download Annotated Video", f, "detection_video.mp4")

                os.unlink(temp_in)
                os.unlink(temp_out)

# ================ PART B: Industrial OCR ================
else:
    st.header("📄 Offline Industrial OCR")
    st.markdown("Extracts **faded/stenciled text** from industrial boxes (military crates, serial numbers, etc.)")

    ocr = load_ocr()

    def ocr_preprocess(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=15)
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    uploaded = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4"], key="ocr")

    if uploaded:
        if uploaded.type.startswith('image'):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                img_pil = Image.open(uploaded)
                st.image(img_pil, use_column_width=True)

            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            preprocessed = ocr_preprocess(img_cv)

            with col2:
                st.subheader("Preprocessed (for OCR)")
                st.image(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB), use_column_width=True)

            with st.spinner("Extracting text..."):
                result = ocr.ocr(preprocessed, cls=True)

            output_img = img_cv.copy()
            extracted = []
            for line in result or []:
                for word in line:
                    bbox = np.array(word[0], np.int32)
                    text, conf = word[1]
                    if conf > conf_threshold:
                        cv2.polylines(output_img, [bbox], True, (0, 255, 0), 3)
                        cv2.putText(output_img, f"{text} ({conf:.2f})", (bbox[0][0], bbox[0][1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        extracted.append({"text": text, "confidence": round(conf, 3)})

            st.subheader("OCR Result")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_column_width=True)

            st.subheader("Extracted Text")
            if extracted:
                st.json(extracted)
                st.download_button("Download JSON", json.dumps(extracted, indent=2), "ocr_results.json")
            else:
                st.info("No text detected above threshold.")

        else:  # Video OCR (every 10th frame)
            st.video(uploaded)
            if st.button("Process Video (OCR)"):
                temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                with open(temp_in, "wb") as f:
                    f.write(uploaded.getbuffer())

                cap = cv2.VideoCapture(temp_in)
                fps = cap.get(cv2.CAP_PROP_FPS)
                w, h = int(cap.get(3)), int(cap.get(4))
                temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                writer = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                progress = st.progress(0)
                frame_count = 0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_count % 10 == 0:
                        pre = ocr_preprocess(frame)
                        res = ocr.ocr(pre, cls=True)
                        for line in res or []:
                            for word in line:
                                if word[1][1] > conf_threshold:
                                    bbox = np.array(word[0], np.int32)
                                    cv2.polylines(frame, [bbox], True, (0, 255, 255), 2)
                                    cv2.putText(frame, word[1][0], (bbox[0][0], bbox[0][1]-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    writer.write(frame)
                    frame_count += 1
                    progress.progress(frame_count / total)

                cap.release()
                writer.release()
                progress.empty()

                st.video(temp_out)
                with open(temp_out, "rb") as f:
                    st.download_button("Download OCR Video", f, "ocr_video.mp4")

                os.unlink(temp_in)
                os.unlink(temp_out)

st.caption("Fully offline • Custom-trained models • No cloud APIs")
