import json
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_PATH = "model.pth"
CLASSES_PATH = "classes.json"


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(model_path: str, device):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes")
    if classes is None and os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = json.load(f)
    if not classes:
        raise ValueError("Classes not found in checkpoint or classes.json")
    model = build_model(len(classes))
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, classes


def predict_single(img_path, model, classes, device, tf):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    return probs


def predict(image_paths, model_path=MODEL_PATH, topk=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(model_path, device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    best_probs = None
    best_image = None
    highest_conf = -1.0

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    for path in image_paths:
        probs = predict_single(path, model, classes, device, tf)
        top1_conf = float(torch.max(probs))
        if top1_conf > highest_conf:
            highest_conf = top1_conf
            best_probs = probs
            best_image = path

    top_probs, top_idx = torch.topk(best_probs, k=min(topk, len(classes)))
    results = [(classes[i], float(p)) for p, i in zip(top_probs, top_idx)]
    return results, best_image

