import csv
import io
import json
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from torchvision import models, transforms

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_PATH = "model.pth"
CLASSES_PATH = "classes.json"
MEDICINE_CSV = "dataset/medicine_info.csv"
TOP_K = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_classes(classes_path: str) -> List[str]:
    try:
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
            print(f"Loaded {len(classes)} classes from '{classes_path}'")
            return classes
    except Exception as e:
        print(f"Error loading '{classes_path}': {e}")
        return []


def load_medicine_info(csv_path: str) -> Dict[str, Dict[str, str]]:
    info: Dict[str, Dict[str, str]] = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("Medicine Name") or "").strip()
                if not name:
                    continue
                info[name.lower()] = {
                    "name": name,
                    "composition": (row.get("Likely Composition / Class") or "").strip(),
                    "usage": (row.get("Usage") or "").strip(),
                    "description": (row.get("Description") or "").strip(),
                    "side_effects": (row.get("Common Side Effects") or "").strip(),
                }
        print(f"Loaded {len(info)} medicine entries from '{csv_path}'")
    except Exception as e:
        print(f"Error loading '{csv_path}': {e}")
    return info


def find_medicine_info(name: str, medicine_info: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not name:
        return None
    lower_name = name.lower().strip()
    if lower_name in medicine_info:
        return medicine_info[lower_name]
    for key, value in medicine_info.items():
        if lower_name in key or key in lower_name:
            return value
    return None


def load_model(model_path: str, num_classes: int):
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model = build_model(num_classes)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(f"Successfully loaded model weights from '{model_path}'")
        return model
    except Exception as e:
        print(f"Error loading model weights from '{model_path}': {e}")
        return None


# Load artifacts on startup
classes = load_classes(CLASSES_PATH)
medicine_info = load_medicine_info(MEDICINE_CSV)
model = load_model(MODEL_PATH, len(classes)) if classes else None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

app = FastAPI(title="Medicine Tablet Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.get("/api/medicines")
async def list_medicine_names():
    """Returns list of medicines for autocomplete/dropdowns."""
    if classes:
        return JSONResponse(content=sorted(classes))
    return JSONResponse(content=[])


@app.post("/predict")
async def predict_endpoint(files: List[UploadFile] = File(...)):
    if not classes or model is None:
        return JSONResponse(status_code=500, content={"error": "Model or classes failed to load."})

    all_predictions = []
    for file in files:
        contents = await file.read()
        if not contents:
            continue

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]

        top_probs, top_idx = torch.topk(probs, k=min(TOP_K, len(classes)))
        results = [
            {"class": classes[i], "probability": float(p)}
            for p, i in zip(top_probs, top_idx)
        ]

        best_class = results[0]["class"] if results else ""
        med_info = find_medicine_info(best_class, medicine_info)

        all_predictions.append({
            "filename": file.filename,
            "results": results,
            "medicine_info": med_info,
        })

    return JSONResponse(content={"predictions": all_predictions})


@app.post("/interactions-from-images")
async def interactions_from_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Handler for image interaction checking."""
    if not classes or model is None:
        return JSONResponse(status_code=500, content={"error": "Model or classes failed to load."})

    async def predict_single(file: UploadFile):
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)
        return classes[top_idx.item()], float(top_prob.item())

    med1_name, med1_conf = await predict_single(file1)
    med2_name, med2_conf = await predict_single(file2)

    return JSONResponse(content={
        "medicine1": med1_name,
        "medicine1_confidence": med1_conf,
        "medicine2": med2_name,
        "medicine2_confidence": med2_conf,
        "found": False,
        "message": "Interaction model disabled in single-model configuration.",
        "interactions": []
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)