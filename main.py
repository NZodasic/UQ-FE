# main.py
import os
import sys
import cv2
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import uvicorn

# Inject Model_Trainer path so we can cleanly reuse the code without rewriting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TRAINER_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Model_Trainer'))
if MODEL_TRAINER_DIR not in sys.path:
    sys.path.append(MODEL_TRAINER_DIR)

from models.resnet_classifier import ResNet50Classifier
from models.uncertainty_wrapper import MCDropoutWrapper
from explainability.gradcam import GradCAM

app = FastAPI(title="UQ-Module Explainable Vision Client")

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- MODEL INITIALIZATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading model on {device}...")

num_classes = 10
model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
try:
    if os.path.exists(MODEL_PATH):
        # We need to map location to our device to avoid CPU env issues if model trained on GPU
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # Determine if the dump is a Dict containing multiple keys (like epoch, model_state, etc)
        # or just the raw parameters Dict directly.
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            try:
                model.load_state_dict(state_dict)
            except:
                # Typically implies DataParallel was used 'module.*'
                cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(cleaned_state_dict)
        print("Model weights loaded successfully.")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}. Using randomly initialized weights.")
except Exception as e:
    print(f"Error loading model weights: {e}. Proceeding with initialized weights for demonstration.")

model = model.to(device)

# Initialize UQ / XAI Wrappers
# Using 15 MC passes for real-time video is a balancing act between exact entropy and FPS
mc_wrapper = MCDropoutWrapper(model, num_samples=15)
cam_layer = model.get_cam_layer()
gradcam = GradCAM(model, cam_layer)

# Class names. We don't have classes.txt parser attached here but we know there are 29.
classes_path = os.path.join(MODEL_TRAINER_DIR, 'train_data', 'classes.txt')
val_classes = []
if os.path.exists(classes_path):
    with open(classes_path, 'r', encoding='utf-8') as f:
        val_classes = [line.strip() for line in f.readlines() if line.strip()]

def get_class_name(idx):
    if idx < len(val_classes):
        return val_classes[idx]
    return f"Class {idx}"

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Pretrained ImageNet stats used by TorchVision ResNet50 weights IMAGENET1K_V2 typically
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# ----------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    file_path = os.path.join(UPLOADS_DIR, video.filename)
    with open(file_path, "wb") as buffer:
        content = await video.read()
        buffer.write(content)
    return JSONResponse(content={"filename": video.filename, "status": "success"})


def process_frame(frame, processing_type: str):
    """Processes a single BGR OpenCV frame according to the requested type using PyTorch Model."""
    h, w, _ = frame.shape
    
    # 1. Transform frame to PyTorch tensor format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = image_transform(rgb_frame).unsqueeze(0).to(device)
    
    if processing_type == "original":
        return frame
        
    if processing_type == "detection":
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            conf_val = conf.item()
            pred_class = pred.item()
            
        class_name = get_class_name(pred_class)
        label = f"{class_name}: {conf_val*100:.1f}%"
        
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        # Bounding box around standard image (since Model lacks object localizer like YOLO)
        cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 255, 0), 2)
        return frame
        
    if processing_type == "uncertainty":
        # 2. Extract MC Dropout metrics
        predictions = mc_wrapper.predict(input_tensor) # [num_samples, batch_size, num_classes]
        _, entropy, variance = mc_wrapper.get_uncertainty_metrics(predictions)
        
        ent_val = entropy[0].item()
        max_var = variance[0].max().item()
        
        # Provide visual feedback of uncertainty using a full-frame color tint interpolation
        tint = np.zeros_like(frame)
        if ent_val > 0.5:
            # High Entropy maps to Red
            tint[:, :, 2] = 255
            alpha = min(0.4, ent_val / 3.0)
        else:
            # Low Entropy maps to Green
            tint[:, :, 1] = 255
            alpha = 0.15
            
        frame = cv2.addWeighted(frame, 1 - alpha, tint, alpha, 0)
        
        cv2.putText(frame, f"Predictive Entropy: {ent_val:.3f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Max Variance: {max_var:.4f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return frame
        
    if processing_type == "explain":
        # 3. Extract Grad-CAM and resize immediately out of the 224x224 tensor space
        cam = gradcam.generate(input_tensor) 
        cam = cv2.resize(cam, (w, h))
        
        heatmap = np.uint8(255 * cam)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        frame = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)
        cv2.putText(frame, "Grad-CAM Heatmap", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    return frame


def generate_frames(video_path: str, processing_type: str):
    if not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    
    # Process frequency depending on hardware configuration.
    # GPU can do this live mostly 1:1, CPU might need frame skipping 1:3 
    frame_skip = 2 if device.type == 'cpu' else 1
    frame_count = 0
    last_processed_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        if frame_count % frame_skip == 0:
            last_processed_frame = process_frame(frame.copy(), processing_type)
        else:
            if last_processed_frame is None:
                last_processed_frame = frame.copy()
        
        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', last_processed_frame)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.get("/video_feed/{filename}/{processing_type}")
async def video_feed(filename: str, processing_type: str):
    file_path = os.path.join(UPLOADS_DIR, filename)
    return StreamingResponse(
        generate_frames(file_path, processing_type), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    print("Starting Explainable AI streaming server at http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
