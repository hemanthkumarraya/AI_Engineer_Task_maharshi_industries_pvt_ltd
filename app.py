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
st.markdown("**Offline Computer Vision Pipeline** – Detection, Classification & OCR")

# ==================== SIDEBAR ====================
st.sidebar.header("Task Selection")
mode = st.sidebar.selectbox(
    "Choose Task",
    ["Object Detection + Human/Animal Classification", "Offline Industrial OCR"]
)

st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.95, 0.5, 0.05)

# ==================== MODEL PATHS ====================
DETECTOR_PATH = "faster_rcnn_object_detector_FAST.pth"
CLASSIFIER_PATH = "animal_classifier.pth"

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_detector():
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, num_classes=2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
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

detector = load_detector()
classifier = load_classifier()
ocr = load_ocr()

classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== HELPER: READ IMAGE WITHOUT ROTATION ====================
def read_image_raw(uploaded_file):
    """
    Read uploaded image using OpenCV (ignores EXIF completely)
    Returns BGR image (OpenCV format) and RGB for display
    """
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)  # Raw pixels, no EXIF
    if img_bgr is None:
        st.error("Could not read image. Unsupported format or corrupted file.")
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb

# ==================== PART A: Detection + Classification ====================
if mode == "Object Detection + Human/Animal Classification":
    st.header("Object Detection + Human/Animal Classification")
    st.markdown("Detects objects → Classifies as **Human** (Green) or **Animal** (Red)")

    uploaded = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"], key="a")

    if uploaded:
        if uploaded.type.startswith("image"):
            # Display original raw image (no rotation)
            st.subheader("Original Image (Exact as Uploaded)")
            _, img_rgb_display = read_image_raw(uploaded)
            st.image(img_rgb_display, use_column_width=True)

            # Process with raw BGR image
            image_bgr, _ = read_image_raw(uploaded)
            if image_bgr is None:
                st.stop()

            # Convert to tensor for model
            img_tensor = transforms.ToTensor()(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)

            with torch.no_grad():
                predictions = detector(img_tensor)[0]

            boxes = predictions['boxes'][predictions['scores'] > conf_threshold].cpu().numpy()
            scores = predictions['scores'][predictions['scores'] > conf_threshold].cpu().numpy()

            output_img = image_bgr.copy()
            crops = []
            results = []

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                crop = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                if crop.size == 0:
                    continue

                crop_tensor = classify_transform(Image.fromarray(crop)).unsqueeze(0)
                with torch.no_grad():
                    output = classifier(crop_tensor)
                pred_idx = output.argmax(1).item()
                label = "Animal" if pred_idx == 0 else "Human"
                conf = torch.softmax(output, dim=1)[0][pred_idx].item()

                crops.append(crop)
                results.append((label, conf))

                color = (0, 0, 255) if label == "Animal" else (0, 255, 0)
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(output_img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

            st.subheader("Result (Annotations on Original Orientation)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_column_width=True)

            if crops:
                st.subheader("Cropped Classifications")
                cols = st.columns(min(len(crops), 4))
                for i, (crop, (label, conf)) in enumerate(zip(crops, results)):
                    with cols[i % 4]:
                        st.image(crop, caption=f"{label} ({conf:.2f})")

        else:  # Video - same logic, raw frames
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
                writer = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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
                    st.download_button("Download Annotated Video", f, "detection_output.mp4")

                os.unlink(temp_in)
                os.unlink(temp_out)

# ==================== PART B: Industrial OCR ====================
else:
    st.header("Offline Industrial OCR")
    st.markdown("Extracts faded/stenciled text from industrial boxes")

    uploaded = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"], key="b")

    if uploaded:
        if uploaded.type.startswith("image"):
            # Original raw display
            st.subheader("Original Image (Exact Orientation)")
            _, img_rgb_display = read_image_raw(uploaded)
            st.image(img_rgb_display, use_column_width=True)

            # Raw BGR for processing
            img_bgr, _ = read_image_raw(uploaded)
            if img_bgr is None:
                st.stop()

            # Preprocessing
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            denoised = cv2.fastNlMeansDenoising(enhanced, h=15)
            binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
            preprocessed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            st.subheader("Preprocessed for OCR")
            st.image(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB), use_column_width=True)

            with st.spinner("Running OCR..."):
                result = ocr.ocr(preprocessed, cls=True)

            output_img = img_bgr.copy()
            extracted = []
            for line in result or []:
                for word in line:
                    if word[1][1] > conf_threshold:
                        bbox = np.array(word[0], np.int32)
                        text, conf = word[1]
                        cv2.polylines(output_img, [bbox], True, (0, 255, 0), 3)
                        cv2.putText(output_img, f"{text} ({conf:.2f})", (bbox[0][0], bbox[0][1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        extracted.append({"text": text, "confidence": round(conf, 3)})

            st.subheader("OCR Result (on Original Orientation)")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_column_width=True)

            if extracted:
                st.subheader("Extracted Text")
                st.json(extracted)
                st.download_button("Download JSON", json.dumps(extracted, indent=2), "ocr_results.json")
            else:
                st.info("No text detected above threshold.")

        else:  # Video OCR - same raw frame reading
            st.video(uploaded)
            if st.button("Process Video (OCR)"):
                temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                with open(temp_in, "wb") as f:
                    f.write(uploaded.getbuffer())

                cap = cv2.VideoCapture(temp_in)
                fps = cap.get(cv2.CAP_PROP_FPS)
                w, h = int(cap.get(3)), int(cap.get(4))
                temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                writer = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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
                    st.download_button("Download OCR Video", f, "ocr_output.mp4")

                os.unlink(temp_in)
                os.unlink(temp_out)

st.caption("Fully offline • Custom-trained models • Original image orientation preserved • CPU compatible")
