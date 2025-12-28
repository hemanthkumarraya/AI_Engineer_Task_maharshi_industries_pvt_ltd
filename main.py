"""
AI Technical Assignment - Complete Offline Processing Pipeline
main.py - Entry point for processing test videos/images

Features:
- Automatically processes two tasks:
  1. Object Detection + Human/Animal Classification (Part A)
  2. Offline Industrial OCR for stenciled text (Part B)
- Input folders:
  - test_videos/object_detection&classification/  → Part A
  - test_videos/OCR/                               → Part B
- Output folders:
  - outputs/object_detection&classification/      → Annotated videos/images (Part A)
  - outputs/OCR/                                  → Annotated + OCR results (Part B)
- Handles both images and videos
- Robust: skips missing folders/files gracefully
- 100% offline after model setup

Author: Hemanth Kumar
Date: December 28, 2025
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from paddleocr import PaddleOCR
import tempfile

# ==================== CONFIGURATION ====================
ROOT_DIR = Path(__file__).parent

INPUT_DETECTION_DIR = ROOT_DIR / "test_videos" / "object_detection&classification"
INPUT_OCR_DIR = ROOT_DIR / "test_videos" / "OCR"

OUTPUT_DETECTION_DIR = ROOT_DIR / "outputs" / "object_detection&classification"
OUTPUT_OCR_DIR = ROOT_DIR / "outputs" / "OCR"

MODEL_DETECTOR = ROOT_DIR / "models" / "faster_rcnn_object_detector_FAST.pth"
MODEL_CLASSIFIER = ROOT_DIR / "models" / "animal_classifier.pth"

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png"}

# Create output directories
OUTPUT_DETECTION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_OCR_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Starting automated processing pipeline...")
print(f"Root directory: {ROOT_DIR}")
print(f"Detection input: {INPUT_DETECTION_DIR}")
print(f"OCR input: {INPUT_OCR_DIR}\n")

# ==================== MODEL LOADING ====================
print("📦 Loading models (offline)...")

@torch.no_grad()
def load_detector():
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, num_classes=2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(MODEL_DETECTOR, map_location="cpu"))
    model.eval()
    return model

@torch.no_grad()
def load_classifier():
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 0: Animal, 1: Human
    model.load_state_dict(torch.load(MODEL_CLASSIFIER, map_location="cpu"))
    model.eval()
    return model

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

print("✅ All models loaded successfully!\n")

# ==================== PART A: Detection + Classification ====================
def process_detection_file(input_path: Path, output_path: Path):
    if input_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        # Image processing
        img = cv2.imread(str(input_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)

        with torch.no_grad():
            preds = detector(img_tensor)[0]

        boxes = preds['boxes'][preds['scores'] > 0.5].cpu().numpy()
        output_img = img.copy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_tensor = classify_transform(Image.fromarray(crop)).unsqueeze(0)
            with torch.no_grad():
                pred = classifier(crop_tensor).argmax(1).item()
            label = "Animal" if pred == 0 else "Human"
            color = (0, 0, 255) if pred == 0 else (0, 255, 0)

            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        cv2.imwrite(str(output_path), output_img)
        print(f"   → Saved: {output_path.name}")

    else:
        # Video processing
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0)

            with torch.no_grad():
                preds = detector(frame_tensor)[0]

            boxes = preds['boxes'][preds['scores'] > 0.5].cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_tensor = classify_transform(Image.fromarray(crop)).unsqueeze(0)
                with torch.no_grad():
                    pred = classifier(crop_tensor).argmax(1).item()
                label = "Animal" if pred == 0 else "Human"
                color = (0, 0, 255) if pred == 0 else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            writer.write(frame)

        cap.release()
        writer.release()
        print(f"   → Saved: {output_path.name}")

# ==================== PART B: Industrial OCR ====================
def preprocess_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=15)
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

def process_ocr_file(input_path: Path, output_path: Path):
    if input_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        img = cv2.imread(str(input_path))
        pre = preprocess_ocr(img)
        result = ocr.ocr(pre, cls=True)

        output_img = img.copy()
        for line in result or []:
            for word in line:
                if word[1][1] > 0.4:
                    bbox = np.array(word[0], np.int32)
                    cv2.polylines(output_img, [bbox], True, (0, 255, 0), 3)
                    cv2.putText(output_img, word[1][0], (bbox[0][0], bbox[0][1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(str(output_path), output_img)
        print(f"   → Saved: {output_path.name}")

    else:
        # Video OCR
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 10 == 0:  # OCR every 10th frame
                pre = preprocess_ocr(frame)
                result = ocr.ocr(pre, cls=True)
                for line in result or []:
                    for word in line:
                        if word[1][1] > 0.4:
                            bbox = np.array(word[0], np.int32)
                            cv2.polylines(frame, [bbox], True, (0, 255, 255), 2)
                            cv2.putText(frame, word[1][0], (bbox[0][0], bbox[0][1]-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        print(f"   → Saved: {output_path.name}")

# ==================== MAIN AUTOMATION ====================
def main():
    print("🔍 PART A: Object Detection + Human/Animal Classification")
    if INPUT_DETECTION_DIR.exists():
        files = [f for f in INPUT_DETECTION_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if files:
            print(f"Found {len(files)} file(s) to process:")
            for file in files:
                out_file = OUTPUT_DETECTION_DIR / file.name.replace(file.suffix, "_annotated" + file.suffix)
                print(f"   Processing: {file.name}")
                process_detection_file(file, out_file)
        else:
            print("   No supported files found (empty or missing). Skipping Part A.")
    else:
        print("   Input folder not found. Skipping Part A.")

    print("\n📄 PART B: Offline Industrial OCR")
    if INPUT_OCR_DIR.exists():
        files = [f for f in INPUT_OCR_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if files:
            print(f"Found {len(files)} file(s) to process:")
            for file in files:
                out_file = OUTPUT_OCR_DIR / file.name.replace(file.suffix, "_ocr" + file.suffix)
                print(f"   Processing: {file.name}")
                process_ocr_file(file, out_file)
        else:
            print("   No supported files found (empty or missing). Skipping Part B.")
    else:
        print("   Input folder not found. Skipping Part B.")

    print("\n🎉 All processing complete! Check 'outputs/' folder.")

if __name__ == "__main__":
    main()
