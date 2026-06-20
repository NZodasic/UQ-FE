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
TRAINING_PROJECT_DIR = os.environ.get("UQ_TRAINING_DIR")
if not TRAINING_PROJECT_DIR:
    for candidate in ['UQ-in-XAI-for-VTSR', 'UQ-in-XAI-for-VTSR-main-win']:
        p = os.path.abspath(os.path.join(BASE_DIR, '..', candidate))
        if os.path.exists(p):
            TRAINING_PROJECT_DIR = p
            break
    if not TRAINING_PROJECT_DIR:
        TRAINING_PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'UQ-in-XAI-for-VTSR'))
TRAINING_PROJECT_DIR = os.path.abspath(TRAINING_PROJECT_DIR)
EXPERIMENTS_DIR = os.path.join(TRAINING_PROJECT_DIR, "EXPERIMENT")
if TRAINING_PROJECT_DIR not in sys.path:
    sys.path.append(TRAINING_PROJECT_DIR)

from models.resnet_classifier import ResNet50Classifier
from models.efficientnet_classifier import EfficientNetB2Classifier
from models.mobilenet_classifier import MobileNetV2Classifier
from models.densenet_classifier import DenseNet121Classifier
from models.uncertainty_wrapper import MCDropoutWrapper
from explainability.gradcam import GradCAM, GradCAMLibraryWrapper, MCGradCAM
from explainability.integrated_gradients import IntegratedGradients
from explainability.saliency import SaliencyMap

class DeterministicWrapper:
    """Fallback wrapper to match MCDropoutWrapper interface but run in standard deterministic eval mode."""
    def __init__(self, model):
        self.model = model

    def predict(self, x: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            probs = torch.softmax(out, dim=1)
        return probs.unsqueeze(0) # [1, batch_size, num_classes]

    def get_uncertainty_metrics(self, predictions: torch.Tensor):
        expected_p = predictions.mean(dim=0) # [batch_size, num_classes]
        entropy = -torch.sum(expected_p * torch.log(expected_p + 1e-12), dim=1)
        variance = torch.zeros_like(expected_p)
        return expected_p, entropy, variance


class TemperatureScaledModel(torch.nn.Module):
    """Applies post-hoc temperature scaling to logits when a run saved temperature.pth."""
    def __init__(self, model, temperature):
        super().__init__()
        self.model = model
        self.temperature = max(float(temperature or 1.0), 0.05)

    def forward(self, x):
        return self.model(x) / self.temperature

    def get_cam_layer(self):
        return self.model.get_cam_layer()


class TTAWrapper:
    """Lightweight test-time augmentation wrapper for predictive dispersion."""
    def __init__(self, model, num_samples=8):
        self.model = model
        self.num_samples = max(1, int(num_samples or 8))
        self.transforms = [
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            for _ in range(self.num_samples)
        ]

    def predict(self, x: torch.Tensor, rgb_frame=None):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            if rgb_frame is None:
                for _ in range(self.num_samples):
                    out = self.model(x)
                    predictions.append(torch.softmax(out, dim=1).unsqueeze(0))
            else:
                batch = torch.stack([transform(rgb_frame) for transform in self.transforms]).to(x.device)
                out = self.model(batch)
                probs = torch.softmax(out, dim=1).unsqueeze(1)
                predictions.append(probs)

        return torch.cat(predictions, dim=0)

    def get_uncertainty_metrics(self, predictions: torch.Tensor):
        expected_p = predictions.mean(dim=0)
        entropy = -torch.sum(expected_p * torch.log(expected_p + 1e-12), dim=1)
        variance = predictions.var(dim=0, unbiased=False)
        return expected_p, entropy, variance


class SmoothGrad:
    """Noise-averaged saliency map that is useful when raw saliency is too brittle."""
    def __init__(self, model, num_samples=12, noise_std=0.08):
        self.model = model
        self.num_samples = max(1, int(num_samples))
        self.noise_std = float(noise_std)

    def generate(self, input_tensor, target_class=None):
        module_states = [(module, module.training) for module in self.model.modules()]
        saliency_maps = []

        try:
            self.model.eval()
            for _ in range(self.num_samples):
                noisy = (input_tensor + torch.randn_like(input_tensor) * self.noise_std).detach()
                noisy.requires_grad_(True)

                with torch.enable_grad():
                    output = self.model(noisy)
                    if target_class is None:
                        selected_class = output.argmax(dim=1).item()
                    else:
                        selected_class = int(target_class)

                    self.model.zero_grad()
                    output[0, selected_class].backward()

                saliency = torch.abs(noisy.grad.detach().cpu()[0]).numpy()
                saliency_maps.append(np.max(saliency, axis=0))
        finally:
            for module, was_training in module_states:
                module.train(was_training)

        saliency = np.mean(np.asarray(saliency_maps), axis=0)
        saliency = saliency - np.min(saliency)
        return saliency / (np.max(saliency) + 1e-8)


class UQFusionCAM:
    """Fuses MC Grad-CAM mean saliency with low-variance reliability weighting."""
    def __init__(self, model, target_layer, cam_method="gradcam", num_samples=8):
        self.mc_cam = MCGradCAM(
            model,
            target_layer,
            cam_method=cam_method,
            num_samples=num_samples,
        )

    def generate(self, input_tensor, target_class=None):
        mean_cam, std_cam = self.mc_cam.generate(input_tensor, target_class)
        std_cam = np.asarray(std_cam, dtype=np.float32)
        std_cam = std_cam - np.min(std_cam)
        reliability = 1.0 - (std_cam / (np.max(std_cam) + 1e-8))
        fused = mean_cam * reliability
        fused = fused - np.min(fused)
        return fused / (np.max(fused) + 1e-8)

    def remove_hooks(self):
        if hasattr(self.mc_cam, "remove_hooks"):
            self.mc_cam.remove_hooks()


UQ_METHOD_OPTIONS = [
    {"id": "deterministic", "label": "Deterministic Entropy"},
    {"id": "mc_dropout", "label": "MC Dropout"},
    {"id": "temperature_scaling", "label": "Temperature Scaling"},
    {"id": "mc_dropout_temperature", "label": "MC Dropout + Temperature"},
    {"id": "tta", "label": "Test-Time Augmentation"},
]

XAI_METHOD_OPTIONS = [
    {"id": "gradcam", "label": "Grad-CAM"},
    {"id": "gradcam++", "label": "Grad-CAM++"},
    {"id": "eigencam", "label": "EigenCAM"},
    {"id": "hirescam", "label": "HiResCAM"},
    {"id": "integrated_gradients", "label": "Integrated Gradients"},
    {"id": "saliency", "label": "Saliency Map"},
    {"id": "smoothgrad", "label": "SmoothGrad"},
    {"id": "uq_gradcam", "label": "UQ + Grad-CAM Fusion"},
]

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
    path = str(path).replace('\\', '/')
    for win_prefix, linux_prefix in [
        ('C:/Users/ezycloudx-admin/Desktop', '/home/raymond/Desktop'),
        ('C:/Users/ezycloudx-admin', '/home/raymond'),
        ('C:/', '/'),
    ]:
        if path.startswith(win_prefix):
            resolved = path.replace(win_prefix, linux_prefix, 1)
            if os.path.exists(resolved):
                return resolved
            for name in ['UQ-in-XAI-for-VTSR-main-win', 'UQ-in-XAI-for-VTSR']:
                if name in resolved:
                    attempt = resolved.replace(name, os.path.basename(TRAINING_PROJECT_DIR))
                    if os.path.exists(attempt):
                        return attempt
            for name in ['UQ-in-XAI-for-VTSR-main-win', 'UQ-in-XAI-for-VTSR']:
                if name in resolved:
                    parts = resolved.split(name + '/')
                    if len(parts) > 1:
                        attempt = os.path.join(TRAINING_PROJECT_DIR, parts[1])
                        if os.path.exists(attempt):
                            return attempt

    if not os.path.isabs(path):
        attempt = os.path.join(TRAINING_PROJECT_DIR, path)
        if os.path.exists(attempt):
            return attempt

    filename = os.path.basename(path)
    for candidate in [
        os.path.join('/home/raymond/Desktop/Dataset/data2-augment', filename),
        os.path.join(TRAINING_PROJECT_DIR, "Dataset", "data2-augment", filename),
        os.path.join(TRAINING_PROJECT_DIR, filename),
    ]:
        if os.path.exists(candidate):
            return candidate

    return path


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
    method, variant = normalize_xai_selection(info.get("xai_method"), info.get("xai_variant"))
    cam_method = variant or method

    if cam_method == "integrated_gradients":
        steps = 24
        config_path = info.get("config_path")
        if config_path:
            config = read_yaml(config_path)
            steps = int(config.get("explainability", {}).get("steps") or steps)
        return IntegratedGradients(model, steps=steps)
    elif cam_method == "saliency":
        return SaliencyMap(model)
    elif cam_method == "smoothgrad":
        return SmoothGrad(model)
    elif cam_method == "uq_gradcam":
        samples = int(info.get("xai_mc_samples") or info.get("uq_samples") or 8)
        return UQFusionCAM(model, model.get_cam_layer(), cam_method="gradcam", num_samples=samples)

    if cam_method in ["gradcam++", "eigencam", "hirescam"]:
        try:
            return GradCAMLibraryWrapper(model, model.get_cam_layer(), method=cam_method)
        except Exception as exc:
            print(f"Failed to load GradCAMLibraryWrapper for {cam_method}: {exc}. Falling back to standard GradCAM.")

    return GradCAM(model, model.get_cam_layer())


def update_bundle_config(bundle, uq_method=None, uq_samples=None, xai_method=None, xai_variant=None):
    info = bundle["info"]

    # Fall back to model's default configs if not explicitly overridden
    if uq_method is None:
        uq_method = info.get("uq_method", "mc_dropout")
    if uq_samples is None:
        uq_samples = info.get("uq_samples", 15)
    if xai_method is None:
        xai_method = info.get("xai_method", "gradcam")
    if xai_variant is None:
        xai_variant = info.get("xai_variant")

    uq_method = normalize_uq_method(uq_method)
    if uq_method in {"temperature_scaling", "mc_dropout_temperature"} and info.get("temperature") is None:
        print(f"Temperature scaling requested for {info['id']} but no temperature value is available. Falling back.")
        uq_method = "mc_dropout" if uq_method == "mc_dropout_temperature" else "deterministic"
    try:
        uq_samples = max(1, int(uq_samples or 1))
    except (TypeError, ValueError):
        uq_samples = 1
    xai_method, xai_variant = normalize_xai_selection(xai_method, xai_variant)

    # Lock and update uncertainty wrapper
    with bundle["uncertainty_lock"]:
        current_uq_method = bundle.get("current_uq_method")
        current_uq_samples = bundle.get("current_uq_samples")
        if (current_uq_method != uq_method) or (current_uq_samples != uq_samples) or ("mc_wrapper" not in bundle):
            print(f"Updating UQ wrapper on model {info['id']}: {uq_method} (samples={uq_samples})")
            use_temperature = uq_method in {"temperature_scaling", "mc_dropout_temperature"}
            uncertainty_model = bundle["temperature_uncertainty_model"] if use_temperature else bundle["uncertainty_model"]

            if uq_method in {"mc_dropout", "mc_dropout_temperature"}:
                bundle["mc_wrapper"] = MCDropoutWrapper(uncertainty_model, num_samples=uq_samples)
            elif uq_method == "tta":
                bundle["mc_wrapper"] = TTAWrapper(uncertainty_model, num_samples=uq_samples)
            else:
                bundle["mc_wrapper"] = DeterministicWrapper(uncertainty_model)
            bundle["current_uq_method"] = uq_method
            bundle["current_uq_samples"] = uq_samples if uq_method in {"mc_dropout", "mc_dropout_temperature", "tta"} else 1

    # Lock and update explainability wrapper
    with bundle["explain_lock"]:
        current_xai_method = bundle.get("current_xai_method")
        current_xai_variant = bundle.get("current_xai_variant")
        if (current_xai_method != xai_method) or (current_xai_variant != xai_variant) or ("explainer" not in bundle):
            print(f"Updating explainer on model {info['id']}: {xai_method} (variant={xai_variant})")
            if "explainer" in bundle and hasattr(bundle["explainer"], "remove_hooks"):
                try:
                    bundle["explainer"].remove_hooks()
                except Exception as e:
                    print(f"Error removing old hooks: {e}")

            temp_info = {
                **info,
                "xai_method": xai_method,
                "xai_variant": xai_variant,
            }
            bundle["explainer"] = build_explainer(temp_info, bundle["explain_model"])
            bundle["current_xai_method"] = xai_method
            bundle["current_xai_variant"] = xai_variant



def model_info_from_run(run_dir):
    config_path = os.path.join(run_dir, "config.yaml")
    model_path = os.path.join(run_dir, "models", "best_model.pth")
    temperature_path = os.path.join(run_dir, "models", "temperature.pth")
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
    calibration_config = config.get("calibration", {})
    temperature = (
        number_or_none(metrics.get("temperature"))
        or number_or_none(metrics_model.get("temperature"))
        or read_temperature_file(temperature_path)
    )
    temperature_available = temperature is not None

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
        "uq_method": normalize_uq_method(
            metrics_uncertainty.get("method") or uncertainty_config.get("method") or "mc_dropout"
        ),
        "uq_samples": int(
            metrics_uncertainty.get("num_samples")
            or uncertainty_config.get("num_samples")
            or 15
        ),
        "xai_method": metrics_explainability.get("method") or explainability_config.get("method") or "gradcam",
        "xai_variant": xai_variant,
        "xai_mc_samples": int(explainability_config.get("mc_samples") or explainability_config.get("num_samples") or 8),
        "temperature_path": temperature_path if os.path.exists(temperature_path) else None,
        "temperature": temperature,
        "temperature_available": temperature_available,
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
        "xai_mc_samples": 8,
        "temperature_path": None,
        "temperature": None,
        "temperature_available": False,
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
        model_root = os.path.join(EXPERIMENTS_DIR, "model")
        if os.path.isdir(model_root):
            for backbone_entry in sorted(os.scandir(model_root), key=lambda item: item.name):
                if not backbone_entry.is_dir():
                    continue
                for run_entry in sorted(os.scandir(backbone_entry.path), key=lambda item: item.name):
                    if not run_entry.is_dir() or not run_entry.name.startswith("run_"):
                        continue
                    info = model_info_from_run(run_entry.path)
                    if info:
                        models.append(info)

        for entry in sorted(os.scandir(EXPERIMENTS_DIR), key=lambda item: item.name):
            if entry.is_dir() and entry.name.startswith("run_"):
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
        "uq_options": make_uq_options(info),
        "xai_options": make_xai_options(info),
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
        temperature = info.get("temperature")
        if temperature:
            temperature_uncertainty_model = TemperatureScaledModel(
                instantiate_model(info, state_dict),
                temperature,
            ).to(device)
        else:
            temperature_uncertainty_model = uncertainty_model

        classes = load_class_names(info.get("classes_file"), info["num_classes"])

        bundle = {
            "info": info,
            "detection_model": detection_model,
            "uncertainty_model": uncertainty_model,
            "temperature_uncertainty_model": temperature_uncertainty_model,
            "explain_model": explain_model,
            "classes": classes,
            "detection_lock": threading.RLock(),
            "uncertainty_lock": threading.RLock(),
            "explain_lock": threading.RLock(),
        }

        # Initialize bundle with model info default settings
        update_bundle_config(
            bundle,
            uq_method=info.get("uq_method"),
            uq_samples=info.get("uq_samples"),
            xai_method=info.get("xai_method"),
            xai_variant=info.get("xai_variant"),
        )

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
async def prediction(
    filename: str,
    model_id: str | None = None,
    uq_method: str | None = None,
    uq_samples: int | None = None,
    xai_method: str | None = None,
    xai_variant: str | None = None,
):
    filename = os.path.basename(filename)
    file_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Uploaded video not found")

    cache_key = analysis_cache_key(filename, model_id, uq_method, uq_samples, xai_method, xai_variant)
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
        metadata = predict_frame_metadata(
            frame,
            model_id=model_id,
            uq_method=uq_method,
            uq_samples=uq_samples,
            xai_method=xai_method,
            xai_variant=xai_variant,
        )
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
        "none": "Deterministic",
        "deterministic": "Deterministic Entropy",
        "mc_dropout": "MC Dropout",
        "temperature_scaling": "Temperature Scaling",
        "mc_dropout_temperature": "MC Dropout + Temperature",
        "tta": "Test-Time Augmentation",
        "gradcam": "Grad-CAM",
        "gradcam++": "Grad-CAM++",
        "eigencam": "EigenCAM",
        "hirescam": "HiResCAM",
        "integrated_gradients": "Integrated Gradients",
        "saliency": "Saliency",
        "smoothgrad": "SmoothGrad",
        "uq_gradcam": "UQ + Grad-CAM Fusion",
    }
    normalized = str(method or "").lower()
    return labels.get(normalized, str(method or "N/A"))


def normalize_uq_method(method):
    normalized = str(method or "mc_dropout").lower()
    aliases = {
        "none": "deterministic",
        "no_uq": "deterministic",
        "deterministic_entropy": "deterministic",
        "temperature": "temperature_scaling",
        "ts": "temperature_scaling",
        "mc_dropout_temperature_scaling": "mc_dropout_temperature",
        "ts_mc_dropout": "mc_dropout_temperature",
    }
    return aliases.get(normalized, normalized)


def normalize_xai_selection(xai_method=None, xai_variant=None):
    method = str(xai_method or "gradcam").lower()
    variant = str(xai_variant).lower() if xai_variant else None

    if method in {"gradcam++", "eigencam", "hirescam"}:
        return "gradcam", method
    if method == "gradcam" and variant in {"gradcam", "none", ""}:
        return "gradcam", None
    if method == "gradcam" and variant:
        return "gradcam", variant
    return method, None


def read_temperature_file(path):
    if not path or not os.path.exists(path):
        return None
    try:
        data = torch_load(path)
        if isinstance(data, dict):
            data = data.get("temperature", data.get("T", data.get("value")))
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().item()
        return number_or_none(data)
    except Exception as exc:
        print(f"Could not load temperature from {path}: {exc}")
        return None


def make_uq_options(info):
    has_temperature = bool(info.get("temperature_available"))
    options = []
    for option in UQ_METHOD_OPTIONS:
        item = {**option}
        if option["id"] in {"temperature_scaling", "mc_dropout_temperature"} and not has_temperature:
            item["disabled"] = True
            item["note"] = "requires temperature.pth"
        options.append(item)
    return options


def make_xai_options(_info=None):
    return [{**option} for option in XAI_METHOD_OPTIONS]


def xai_display_name(model_info):
    method = str(model_info.get("xai_method") or "").lower()
    variant = model_info.get("xai_variant")
    if method == "gradcam" and variant:
        return display_method_name(variant)
    return display_method_name(method or variant)


def uq_display_name(method, samples=None, temperature=None):
    method = normalize_uq_method(method)
    label = display_method_name(method)
    details = []
    if method in {"mc_dropout", "mc_dropout_temperature", "tta"}:
        details.append(f"{int(samples or 1)} samples")
    if method in {"temperature_scaling", "mc_dropout_temperature"} and temperature:
        details.append(f"T={float(temperature):.2f}")
    return f"{label} ({', '.join(details)})" if details else label


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


def uncertainty_metrics(bundle, input_tensor, rgb_frame=None):
    with bundle["uncertainty_lock"]:
        if isinstance(bundle["mc_wrapper"], TTAWrapper):
            predictions = bundle["mc_wrapper"].predict(input_tensor, rgb_frame=rgb_frame)
        else:
            predictions = bundle["mc_wrapper"].predict(input_tensor)
        expected_p, entropy, variance = bundle["mc_wrapper"].get_uncertainty_metrics(predictions)

    expected = expected_p[0]
    top_prob, top_class = torch.max(expected, dim=0)
    method = bundle.get("current_uq_method", bundle["info"].get("uq_method"))
    samples = int(bundle.get("current_uq_samples") or 1)
    return {
        "method": display_method_name(method),
        "method_id": method,
        "samples": samples,
        "temperature": bundle["info"].get("temperature") if method in {"temperature_scaling", "mc_dropout_temperature"} else None,
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
        "method": xai_display_name({
            "xai_method": bundle.get("current_xai_method"),
            "xai_variant": bundle.get("current_xai_variant"),
        }),
        "method_id": bundle.get("current_xai_variant") or bundle.get("current_xai_method"),
        "peak_saliency": float(np.max(heatmap)),
        "mean_saliency": float(np.mean(heatmap)),
        "active_area": float(np.count_nonzero(active_mask) / total_pixels),
        "strong_area": float(np.count_nonzero(strong_mask) / total_pixels),
        "focus_ratio": float(np.mean(heatmap[active_mask]) if np.any(active_mask) else 0.0),
        "center_x": center_x,
        "center_y": center_y,
    }


def get_dynamic_stream_settings(processing_type, uq_samples, is_cpu):
    if processing_type == "original":
        return 1, 1 / 24, 85

    if processing_type == "detection":
        skip = 3 if is_cpu else 1
        delay = 1 / 12 if is_cpu else 1 / 18
        return skip, delay, 80

    if processing_type == "uncertainty":
        if uq_samples <= 1:
            # Deterministic prediction is fast
            skip = 3 if is_cpu else 1
            delay = 1 / 12 if is_cpu else 1 / 18
        else:
            # MC Dropout is slower
            skip = 14 if is_cpu else 4
            delay = 1 / 3 if is_cpu else 1 / 8
        return skip, delay, 72

    if processing_type == "explain":
        # Explainability is compute intensive
        skip = 18 if is_cpu else 5
        delay = 1 / 2 if is_cpu else 1 / 6
        return skip, delay, 72

    return 3, 1 / 10, 78


def predict_frame_metadata(
    frame,
    model_id=None,
    uq_method=None,
    uq_samples=None,
    xai_method=None,
    xai_variant=None,
):
    bundle = load_model_bundle(model_id)
    update_bundle_config(bundle, uq_method, uq_samples, xai_method, xai_variant)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = image_transform(rgb_frame).unsqueeze(0).to(device)
    prediction = predict_input_tensor(bundle, input_tensor)
    model_info = bundle["info"]
    uncertainty = uncertainty_metrics(bundle, input_tensor, rgb_frame=rgb_frame)
    heatmap = generate_explanation_map(bundle, input_tensor, target_class=prediction["class_index"])
    explanation = explanation_metrics(bundle, heatmap)

    return {
        **prediction,
        "model_id": model_info["id"],
        "model_label": model_info["model_label"],
        "num_classes": model_info["num_classes"],
        "uq_label": uq_display_name(
            bundle.get("current_uq_method"),
            bundle.get("current_uq_samples"),
            model_info.get("temperature"),
        ),
        "xai_label": xai_display_name({
            "xai_method": bundle.get("current_xai_method"),
            "xai_variant": bundle.get("current_xai_variant"),
        }),
        "uncertainty": uncertainty,
        "explanation": explanation,
    }


def analysis_cache_key(filename, model_id, uq_method=None, uq_samples=None, xai_method=None, xai_variant=None):
    return f"{os.path.basename(filename)}::{model_id or get_active_model_id()}::{uq_method}::{uq_samples}::{xai_method}::{xai_variant}"


def process_frame(
    frame,
    processing_type: str,
    model_id=None,
    uq_method=None,
    uq_samples=None,
    xai_method=None,
    xai_variant=None,
):
    """Processes a single BGR OpenCV frame according to the requested type using PyTorch Model."""
    h, w, _ = frame.shape

    if processing_type == "original":
        return frame

    try:
        bundle = load_model_bundle(model_id)
        update_bundle_config(bundle, uq_method, uq_samples, xai_method, xai_variant)
    except Exception as exc:
        print(f"Model load or configuration failed: {exc}")
        return put_status(frame, f"Model load failed: {exc}", (0, 0, 255))

    model_info = bundle["info"]

    # 1. Transform frame to PyTorch tensor format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = image_transform(rgb_frame).unsqueeze(0).to(device)

    if processing_type == "detection":
        prediction = predict_input_tensor(bundle, input_tensor)
        label = f"{prediction['class_name']}: {prediction['confidence_label']}"

        uq_label = uq_display_name(
            bundle.get("current_uq_method"),
            bundle.get("current_uq_samples"),
            model_info.get("temperature"),
        )
        xai_lbl = xai_display_name({
            "xai_method": bundle.get("current_xai_method"),
            "xai_variant": bundle.get("current_xai_variant"),
        })
        draw_detection_panel(frame, [
            f"Predicted Class: {label}",
            f"Model: {model_info['model_label']}",
            f"UQ: {uq_label} | XAI: {xai_lbl}",
        ])
        cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 255, 0), 2)
        return frame

    if processing_type == "uncertainty":
        metrics = uncertainty_metrics(bundle, input_tensor, rgb_frame=rgb_frame)
        ent_val = metrics["predictive_entropy"]
        max_var = metrics["max_variance"]
        uq_meth = bundle.get("current_uq_method", "mc_dropout")

        tint = np.zeros_like(frame)
        if uq_meth == "deterministic":
            tint[:, :, 1] = 255
            alpha = 0.10
        elif ent_val > 0.5:
            tint[:, :, 2] = 255
            alpha = min(0.4, ent_val / 3.0)
        else:
            tint[:, :, 1] = 255
            alpha = 0.15

        frame = cv2.addWeighted(frame, 1 - alpha, tint, alpha, 0)

        cv2.putText(frame, f"UQ Method: {display_method_name(uq_meth)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if uq_meth != "deterministic":
            cv2.putText(frame, f"Predictive Entropy: {ent_val:.3f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Max Variance: {max_var:.4f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"Entropy: {ent_val:.3f} | variance disabled", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return frame

    if processing_type == "explain":
        prediction = predict_input_tensor(bundle, input_tensor)
        cam = generate_explanation_map(bundle, input_tensor, target_class=prediction["class_index"])
        cam = cv2.resize(cam, (w, h))

        heatmap = np.uint8(255 * cam)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        frame = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)

        xai_lbl = xai_display_name({
            "xai_method": bundle.get("current_xai_method"),
            "xai_variant": bundle.get("current_xai_variant"),
        })
        cv2.putText(frame, f"{xai_lbl} Heatmap", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    return frame


def generate_frames(
    video_path: str,
    processing_type: str,
    model_id=None,
    uq_method=None,
    uq_samples=None,
    xai_method=None,
    xai_variant=None,
):
    if not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    last_processed_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        try:
            bundle = load_model_bundle(model_id)
            update_bundle_config(bundle, uq_method, uq_samples, xai_method, xai_variant)
            current_samples = bundle.get("current_uq_samples", 15)
        except Exception:
            current_samples = 15

        frame_skip, delay, quality = get_dynamic_stream_settings(processing_type, current_samples, device.type == "cpu")
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        if frame_count % frame_skip == 0:
            last_processed_frame = process_frame(
                frame.copy(),
                processing_type,
                model_id=model_id,
                uq_method=uq_method,
                uq_samples=uq_samples,
                xai_method=xai_method,
                xai_variant=xai_variant,
            )
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
async def video_feed(
    filename: str,
    processing_type: str,
    model_id: str | None = None,
    uq_method: str | None = None,
    uq_samples: int | None = None,
    xai_method: str | None = None,
    xai_variant: str | None = None,
):
    filename = os.path.basename(filename)
    file_path = os.path.join(UPLOADS_DIR, filename)
    return StreamingResponse(
        generate_frames(
            file_path,
            processing_type,
            model_id=model_id,
            uq_method=uq_method,
            uq_samples=uq_samples,
            xai_method=xai_method,
            xai_variant=xai_variant,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    print("Starting Explainable AI streaming server at http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
