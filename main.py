# main.py
import json
import os
import sys
import cv2
import time
import torch
import threading
import numpy as np
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import uvicorn
import yaml

# Inject the training project path so this frontend can reuse trained model code.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_PROJECT_DIR = os.environ.get(
    "UQ_TRAINING_DIR",
    os.path.abspath(os.path.join(BASE_DIR, '..', 'UQ-in-XAI-for-VTSR-main-win'))
)
TRAINING_PROJECT_DIR = os.path.abspath(TRAINING_PROJECT_DIR)
EXPERIMENTS_DIR = os.path.join(TRAINING_PROJECT_DIR, "EXPERIMENT")
if TRAINING_PROJECT_DIR not in sys.path:
    sys.path.append(TRAINING_PROJECT_DIR)

from models.resnet_classifier import ResNet50Classifier
from models.efficientnet_classifier import EfficientNetB2Classifier
from models.mobilenet_classifier import MobileNetV2Classifier
from models.densenet_classifier import DenseNet121Classifier
from models.uncertainty_wrapper import MCDropoutWrapper
from explainability.gradcam import GradCAM
from explainability.integrated_gradients import IntegratedGradients

app = FastAPI(title="UQ-Module Explainable Vision Client")

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- MODEL INITIALIZATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Model device: {device}")

MODEL_BUILDERS = {
    "resnet50": ResNet50Classifier,
    "efficientnet_b2": EfficientNetB2Classifier,
    "mobilenet_v2": MobileNetV2Classifier,
    "densenet121": DenseNet121Classifier,
}

model_lock = threading.RLock()
model_cache = {}
analysis_cache = {}
active_model_id = None


def processing_frame_skip(processing_type):
    if device.type != "cpu":
        return {
            "original": 1,
            "detection": 1,
            "uncertainty": 4,
            "explain": 5,
        }.get(processing_type, 1)

    return {
        "original": 1,
        "detection": 3,
        "uncertainty": 14,
        "explain": 18,
    }.get(processing_type, 3)


def stream_delay(processing_type):
    if device.type != "cpu":
        return {
            "original": 1 / 24,
            "detection": 1 / 18,
            "uncertainty": 1 / 8,
            "explain": 1 / 6,
        }.get(processing_type, 1 / 18)

    return {
        "original": 1 / 24,
        "detection": 1 / 12,
        "uncertainty": 1 / 3,
        "explain": 1 / 2,
    }.get(processing_type, 1 / 10)


def jpeg_quality(processing_type):
    return {
        "original": 82,
        "detection": 80,
        "uncertainty": 72,
        "explain": 72,
    }.get(processing_type, 78)


def read_yaml(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Could not read YAML {path}: {exc}")
        return {}


def read_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Could not read JSON {path}: {exc}")
        return {}


def resolve_training_path(path):
    if not path:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.join(TRAINING_PROJECT_DIR, path)


def number_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_percent(value):
    value = number_or_none(value)
    if value is None:
        return None
    return f"{value * 100:.2f}%"


def model_label(model_name):
    labels = {
        "resnet50": "ResNet50",
        "efficientnet_b2": "EfficientNet-B2",
        "mobilenet_v2": "MobileNetV2",
        "densenet121": "DenseNet121",
    }
    return labels.get(model_name, model_name or "Unknown model")


def load_class_names(classes_file, num_classes):
    class_names = []
    candidates = []
    resolved = resolve_training_path(classes_file)
    if resolved:
        candidates.append(resolved)
    candidates.extend([
        os.path.join(TRAINING_PROJECT_DIR, "Dataset", "data2-augment", "custom_data.yaml"),
        os.path.join(TRAINING_PROJECT_DIR, "Dataset", "data2-augment", "classes.txt"),
    ])

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            if path.endswith((".yaml", ".yml")):
                data = read_yaml(path)
                names = data.get("names", [])
                if isinstance(names, dict):
                    class_names = [
                        str(names[key])
                        for key in sorted(names, key=lambda item: int(item))
                    ]
                elif isinstance(names, list):
                    class_names = [str(name) for name in names]
            else:
                with open(path, "r", encoding="utf-8") as f:
                    class_names = [line.strip() for line in f if line.strip()]
        except Exception as exc:
            print(f"Could not read class names from {path}: {exc}")
            class_names = []

        if class_names:
            return class_names

    return [f"Class {idx}" for idx in range(num_classes)]


def get_class_name(classes, idx):
    if idx < len(classes):
        return classes[idx]
    return f"Class {idx}"


def build_model(model_name, num_classes, dropout_rate):
    normalized = (model_name or "efficientnet_b2").lower()
    if normalized not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    builder = MODEL_BUILDERS[normalized]
    return builder(
        num_classes=num_classes,
        pretrained=False,
        dropout_rate=dropout_rate,
    )


def torch_load(path):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint does not contain a state dictionary")

    return {
        key.replace("module.", "", 1): value
        for key, value in checkpoint.items()
    }


def instantiate_model(info, state_dict):
    model = build_model(
        info["model_name"],
        info["num_classes"],
        info["dropout_rate"],
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def build_explainer(info, model):
    method = str(info.get("xai_method") or "").lower()
    if method == "integrated_gradients":
        steps = 24
        config_path = info.get("config_path")
        if config_path:
            config = read_yaml(config_path)
            steps = int(config.get("explainability", {}).get("steps") or steps)
        return IntegratedGradients(model, steps=steps)
    return GradCAM(model, model.get_cam_layer())


def model_info_from_run(run_dir):
    config_path = os.path.join(run_dir, "config.yaml")
    model_path = os.path.join(run_dir, "models", "best_model.pth")
    if not os.path.exists(model_path):
        return None

    config = read_yaml(config_path)
    metrics = read_json(os.path.join(run_dir, "metrics_summary.json"))
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    uncertainty_config = config.get("uncertainty", {})
    explainability_config = config.get("explainability", {})
    metrics_model = metrics.get("model", {})
    metrics_training = metrics.get("training", {})
    metrics_classification = metrics.get("classification", {})
    metrics_uncertainty = metrics.get("uncertainty", {})
    metrics_explainability = metrics.get("explainability", {})

    model_name = model_config.get("name") or metrics_model.get("name") or "efficientnet_b2"
    run_id = os.path.basename(run_dir)
    xai_variant = (
        metrics_explainability.get("variant")
        or explainability_config.get("variant")
        or metrics_explainability.get("method")
        or explainability_config.get("method")
    )
    method = xai_variant or metrics_explainability.get("method") or explainability_config.get("method") or "trained"
    display_label = f"{model_label(model_name)} | {method}"

    return {
        "id": run_id,
        "label": display_label,
        "source": "training",
        "run_dir": run_dir,
        "path": model_path,
        "config_path": config_path if os.path.exists(config_path) else None,
        "metrics_path": os.path.join(run_dir, "metrics_summary.json"),
        "model_name": model_name,
        "model_label": model_label(model_name),
        "num_classes": int(data_config.get("num_classes") or 29),
        "dropout_rate": float(model_config.get("dropout_rate") or 0.4),
        "classes_file": data_config.get("classes_file"),
        "uq_method": metrics_uncertainty.get("method") or uncertainty_config.get("method") or "mc_dropout",
        "uq_samples": int(
            metrics_uncertainty.get("num_samples")
            or uncertainty_config.get("num_samples")
            or 15
        ),
        "xai_method": metrics_explainability.get("method") or explainability_config.get("method") or "gradcam",
        "xai_variant": xai_variant,
        "accuracy": number_or_none(metrics_classification.get("accuracy")),
        "f1": number_or_none(metrics_classification.get("f1")),
        "best_val_accuracy": number_or_none(metrics_training.get("best_val_accuracy")),
        "latency_ms": number_or_none(metrics.get("latency_ms")),
        "size_mb": number_or_none(metrics_model.get("size_mb")),
        "mtime": os.path.getmtime(model_path),
    }


def local_model_info():
    if not os.path.exists(LOCAL_MODEL_PATH):
        return None

    return {
        "id": "local-best-model",
        "label": "ResNet50 | local",
        "source": "local",
        "run_dir": None,
        "path": LOCAL_MODEL_PATH,
        "config_path": None,
        "metrics_path": None,
        "model_name": "resnet50",
        "model_label": "ResNet50",
        "num_classes": 10,
        "dropout_rate": 0.5,
        "classes_file": None,
        "uq_method": "mc_dropout",
        "uq_samples": 15,
        "xai_method": "gradcam",
        "xai_variant": "gradcam",
        "accuracy": None,
        "f1": None,
        "best_val_accuracy": None,
        "latency_ms": None,
        "size_mb": os.path.getsize(LOCAL_MODEL_PATH) / (1024 * 1024),
        "mtime": os.path.getmtime(LOCAL_MODEL_PATH),
    }


def discover_models():
    models = []
    if os.path.isdir(EXPERIMENTS_DIR):
        for entry in sorted(os.scandir(EXPERIMENTS_DIR), key=lambda item: item.name):
            if not entry.is_dir() or not entry.name.startswith("run_"):
                continue
            info = model_info_from_run(entry.path)
            if info:
                models.append(info)

    local_info = local_model_info()
    if local_info:
        models.append(local_info)

    models.sort(key=lambda item: (item["source"] != "training", -item["mtime"]))
    return models


def public_model_info(info):
    return {
        **{key: value for key, value in info.items() if key != "path"},
        "path": os.path.relpath(info["path"], BASE_DIR),
        "accuracy_label": format_percent(info.get("accuracy")),
        "f1_label": format_percent(info.get("f1")),
        "best_val_accuracy_label": format_percent(info.get("best_val_accuracy")),
    }


def get_model_info(model_id=None):
    models = discover_models()
    if not models:
        raise HTTPException(
            status_code=404,
            detail=f"No model checkpoints found in {EXPERIMENTS_DIR} or {LOCAL_MODEL_PATH}",
        )

    selected_id = model_id or get_active_model_id(models)
    for info in models:
        if info["id"] == selected_id:
            return info

    raise HTTPException(status_code=404, detail=f"Unknown model id: {selected_id}")


def get_active_model_id(models=None):
    global active_model_id
    models = models or discover_models()
    model_ids = {info["id"] for info in models}

    with model_lock:
        if active_model_id not in model_ids:
            active_model_id = models[0]["id"] if models else None
        return active_model_id


def set_active_model_id(model_id):
    global active_model_id
    info = get_model_info(model_id)
    with model_lock:
        active_model_id = info["id"]
    return info


def load_model_bundle(model_id=None):
    info = get_model_info(model_id)
    cache_key = info["id"]

    with model_lock:
        cached = model_cache.get(cache_key)
        if cached:
            return cached

        print(f"Loading {info['label']} from {info['path']} on {device}...")
        checkpoint = torch_load(info["path"])
        state_dict = extract_state_dict(checkpoint)
        detection_model = instantiate_model(info, state_dict)
        uncertainty_model = instantiate_model(info, state_dict)
        explain_model = instantiate_model(info, state_dict)

        classes = load_class_names(info.get("classes_file"), info["num_classes"])
        mc_wrapper = MCDropoutWrapper(uncertainty_model, num_samples=info["uq_samples"])
        explainer = build_explainer(info, explain_model)

        bundle = {
            "info": info,
            "detection_model": detection_model,
            "uncertainty_model": uncertainty_model,
            "explain_model": explain_model,
            "classes": classes,
            "mc_wrapper": mc_wrapper,
            "explainer": explainer,
            "detection_lock": threading.RLock(),
            "uncertainty_lock": threading.RLock(),
            "explain_lock": threading.RLock(),
        }
        model_cache[cache_key] = bundle
        print(f"Model loaded: {info['label']}")
        return bundle

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
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/models")
async def list_models():
    models = discover_models()
    return JSONResponse(content={
        "active_model_id": get_active_model_id(models),
        "training_dir": TRAINING_PROJECT_DIR,
        "experiment_dir": EXPERIMENTS_DIR,
        "models": [public_model_info(info) for info in models],
    })


@app.post("/models/{model_id}/select")
async def select_model(model_id: str):
    # Validate the selected checkpoint immediately so the UI can show failures early.
    bundle = load_model_bundle(model_id)
    info = set_active_model_id(model_id)
    return JSONResponse(content={
        "active_model_id": bundle["info"]["id"],
        "model": public_model_info(bundle["info"]),
    })


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    filename = os.path.basename(video.filename)
    if not filename:
        raise HTTPException(status_code=400, detail="Missing upload filename")

    file_path = os.path.join(UPLOADS_DIR, filename)
    with open(file_path, "wb") as buffer:
        content = await video.read()
        buffer.write(content)

    with model_lock:
        stale_keys = [key for key in analysis_cache if key.startswith(f"{filename}::")]
        for key in stale_keys:
            analysis_cache.pop(key, None)

    return JSONResponse(content={"filename": filename, "status": "success"})


@app.get("/prediction/{filename}")
async def prediction(filename: str, model_id: str | None = None):
    filename = os.path.basename(filename)
    file_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Uploaded video not found")

    cache_key = analysis_cache_key(filename, model_id)
    with model_lock:
        cached = analysis_cache.get(cache_key)
    if cached:
        return JSONResponse(content=cached)

    cap = cv2.VideoCapture(file_path)
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 3))
        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret:
        raise HTTPException(status_code=422, detail="Could not read a frame from the uploaded video")

    try:
        metadata = predict_frame_metadata(frame, model_id)
    except Exception as exc:
        print(f"Prediction metadata failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with model_lock:
        analysis_cache[cache_key] = metadata

    return JSONResponse(content=metadata)


def put_status(frame, message, color=(255, 255, 255)):
    cv2.putText(frame, message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def display_method_name(method):
    labels = {
        "mc_dropout": "MC Dropout",
        "gradcam": "Grad-CAM",
        "gradcam++": "Grad-CAM++",
        "eigencam": "EigenCAM",
        "hirescam": "HiResCAM",
        "integrated_gradients": "Integrated Gradients",
        "saliency": "Saliency",
    }
    normalized = str(method or "").lower()
    return labels.get(normalized, str(method or "N/A"))


def xai_display_name(model_info):
    method = str(model_info.get("xai_method") or "").lower()
    variant = model_info.get("xai_variant")
    if method == "gradcam" and variant:
        return display_method_name(variant)
    return display_method_name(method or variant)


def fit_text_to_width(text, font, scale, thickness, max_width):
    text = str(text)
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text

    ellipsis = "..."
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = text[:mid].rstrip() + ellipsis
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if width <= max_width:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip() + ellipsis


def draw_detection_panel(frame, lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.68
    thickness = 2
    padding_x = 14
    padding_y = 10
    line_gap = 8
    max_text_width = max(120, frame.shape[1] - 48)
    fitted_lines = [
        fit_text_to_width(line, font, scale, thickness, max_text_width)
        for line in lines
    ]
    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in fitted_lines]
    box_width = max(width for width, _ in text_sizes) + padding_x * 2
    line_height = max(height for _, height in text_sizes)
    box_height = len(fitted_lines) * line_height + (len(fitted_lines) - 1) * line_gap + padding_y * 2

    x1, y1 = 18, 18
    x2, y2 = x1 + box_width, y1 + box_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (3, 7, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)

    y = y1 + padding_y + line_height
    for index, line in enumerate(fitted_lines):
        color = (52, 255, 121) if index == 0 else (245, 248, 255)
        cv2.putText(frame, line, (x1 + padding_x, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_height + line_gap


def predict_input_tensor(bundle, input_tensor):
    model = bundle["detection_model"]
    with bundle["detection_lock"]:
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

    pred_class = pred.item()
    conf_val = conf.item()
    class_name = get_class_name(bundle["classes"], pred_class)
    return {
        "class_index": pred_class,
        "class_name": class_name,
        "confidence": conf_val,
        "confidence_label": f"{conf_val * 100:.1f}%",
    }


def uncertainty_metrics(bundle, input_tensor):
    with bundle["uncertainty_lock"]:
        predictions = bundle["mc_wrapper"].predict(input_tensor)
        expected_p, entropy, variance = bundle["mc_wrapper"].get_uncertainty_metrics(predictions)

    expected = expected_p[0]
    top_prob, top_class = torch.max(expected, dim=0)
    return {
        "method": display_method_name(bundle["info"].get("uq_method")),
        "samples": int(bundle["info"].get("uq_samples") or 0),
        "predictive_entropy": float(entropy[0].item()),
        "max_variance": float(variance[0].max().item()),
        "mean_variance": float(variance[0].mean().item()),
        "top_probability": float(top_prob.item()),
        "top_class_index": int(top_class.item()),
        "top_class_name": get_class_name(bundle["classes"], int(top_class.item())),
    }


def generate_explanation_map(bundle, input_tensor, target_class=None):
    with bundle["explain_lock"]:
        bundle["explain_model"].eval()
        return bundle["explainer"].generate(input_tensor, target_class)


def explanation_metrics(bundle, heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    active_mask = heatmap >= 0.5
    strong_mask = heatmap >= 0.75
    total_pixels = max(1, heatmap.size)
    total_energy = float(np.sum(heatmap))

    if total_energy > 0:
        ys, xs = np.indices(heatmap.shape)
        center_x = float(np.sum(xs * heatmap) / total_energy)
        center_y = float(np.sum(ys * heatmap) / total_energy)
    else:
        center_x = center_y = 0.0

    return {
        "method": xai_display_name(bundle["info"]),
        "peak_saliency": float(np.max(heatmap)),
        "mean_saliency": float(np.mean(heatmap)),
        "active_area": float(np.count_nonzero(active_mask) / total_pixels),
        "strong_area": float(np.count_nonzero(strong_mask) / total_pixels),
        "focus_ratio": float(np.mean(heatmap[active_mask]) if np.any(active_mask) else 0.0),
        "center_x": center_x,
        "center_y": center_y,
    }


def predict_frame_metadata(frame, model_id=None):
    bundle = load_model_bundle(model_id)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = image_transform(rgb_frame).unsqueeze(0).to(device)
    prediction = predict_input_tensor(bundle, input_tensor)
    model_info = bundle["info"]
    uncertainty = uncertainty_metrics(bundle, input_tensor)
    heatmap = generate_explanation_map(bundle, input_tensor, target_class=prediction["class_index"])
    explanation = explanation_metrics(bundle, heatmap)

    return {
        **prediction,
        "model_id": model_info["id"],
        "model_label": model_info["model_label"],
        "num_classes": model_info["num_classes"],
        "uq_label": f"{display_method_name(model_info.get('uq_method'))} ({model_info.get('uq_samples')})",
        "xai_label": xai_display_name(model_info),
        "uncertainty": uncertainty,
        "explanation": explanation,
    }


def analysis_cache_key(filename, model_id):
    return f"{os.path.basename(filename)}::{model_id or get_active_model_id()}"


def process_frame(frame, processing_type: str, model_id=None):
    """Processes a single BGR OpenCV frame according to the requested type using PyTorch Model."""
    h, w, _ = frame.shape

    if processing_type == "original":
        return frame

    try:
        bundle = load_model_bundle(model_id)
    except Exception as exc:
        print(f"Model load failed: {exc}")
        return put_status(frame, f"Model load failed: {exc}", (0, 0, 255))

    model_info = bundle["info"]

    # 1. Transform frame to PyTorch tensor format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = image_transform(rgb_frame).unsqueeze(0).to(device)

    if processing_type == "detection":
        prediction = predict_input_tensor(bundle, input_tensor)
        label = f"{prediction['class_name']}: {prediction['confidence_label']}"

        uq_label = f"{display_method_name(model_info.get('uq_method'))} ({model_info.get('uq_samples')})"
        draw_detection_panel(frame, [
            f"Predicted Class: {label}",
            f"Model: {model_info['model_label']}",
            f"UQ: {uq_label} | XAI: {xai_display_name(model_info)}",
        ])
        # Bounding box around standard image (since Model lacks object localizer like YOLO)
        cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 255, 0), 2)
        return frame
        
    if processing_type == "uncertainty":
        # 2. Extract MC Dropout metrics
        metrics = uncertainty_metrics(bundle, input_tensor)
        ent_val = metrics["predictive_entropy"]
        max_var = metrics["max_variance"]
        
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
        prediction = predict_input_tensor(bundle, input_tensor)
        cam = generate_explanation_map(bundle, input_tensor, target_class=prediction["class_index"])
        cam = cv2.resize(cam, (w, h))
        
        heatmap = np.uint8(255 * cam)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        frame = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)
        cv2.putText(frame, f"{xai_display_name(model_info)} Heatmap", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    return frame


def generate_frames(video_path: str, processing_type: str, model_id=None):
    if not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)

    frame_skip = processing_frame_skip(processing_type)
    delay = stream_delay(processing_type)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality(processing_type)]
    frame_count = 0
    last_processed_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        if frame_count % frame_skip == 0:
            last_processed_frame = process_frame(frame.copy(), processing_type, model_id)
        else:
            if last_processed_frame is None:
                last_processed_frame = frame.copy()
        
        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', last_processed_frame, encode_params)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(delay)

    cap.release()

@app.get("/video_feed/{filename}/{processing_type}")
async def video_feed(filename: str, processing_type: str, model_id: str | None = None):
    filename = os.path.basename(filename)
    file_path = os.path.join(UPLOADS_DIR, filename)
    return StreamingResponse(
        generate_frames(file_path, processing_type, model_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    print("Starting Explainable AI streaming server at http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
