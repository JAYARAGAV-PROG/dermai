"""
DermAI — FastAPI ML Backend
────────────────────────────────────────────────────────────
Serves the trained EfficientNet-B4 model as a REST API.
The React frontend calls POST /predict with an image file.
Returns real probability scores per skin lesion class.

Run locally:
  uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""

import io, json, time
from pathlib import Path

import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/dermai_model_full.pth")
META_PATH  = Path("models/model_meta.json")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 380

# ── Model (must match training code exactly) ──────────────────────────────────
class DermAIClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0, global_pool=""
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        return self.fc(self.dropout(self.pool(f).flatten(1)))

# ── Preprocessing (must match val_transforms from training) ───────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Class metadata (fallback if model file not found yet) ─────────────────────
DEFAULT_META = {
    "classes":           ["akiec","bcc","bkl","df","mel","nv","vasc"],
    "class_names": {
        "nv":    "Melanocytic Nevi",
        "mel":   "Melanoma",
        "bkl":   "Benign Keratosis",
        "bcc":   "Basal Cell Carcinoma",
        "akiec": "Actinic Keratosis",
        "vasc":  "Vascular Lesion",
        "df":    "Dermatofibroma"
    },
    "malignant_classes": ["mel","bcc","akiec"],
}

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    if not MODEL_PATH.exists():
        return None, DEFAULT_META

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    meta = {
        "classes":           ckpt.get("classes",           DEFAULT_META["classes"]),
        "class_names":       ckpt.get("class_names",       DEFAULT_META["class_names"]),
        "malignant_classes": ckpt.get("malignant_classes", DEFAULT_META["malignant_classes"]),
        "idx_to_class":      {int(k): v for k, v in ckpt.get("idx_to_class", {}).items()},
    }
    model = DermAIClassifier(num_classes=len(meta["classes"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, meta

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="DermAI ML Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production: set to your Netlify URL
    allow_methods=["*"],
    allow_headers=["*"],
)

model, class_meta = None, DEFAULT_META

@app.on_event("startup")
async def startup():
    global model, class_meta
    print(f"Loading model on {DEVICE}...")
    model, class_meta = load_model()
    if model:
        print(f"[OK] Model loaded -- classes: {class_meta['classes']}")
    else:
        print("[WARN] Model file not found -- running in DEMO mode")
        print("   Train the model using the Colab notebook first.")
        print(f"   Then place dermai_model_full.pth in: {MODEL_PATH.parent.absolute()}")

# ── Schemas ───────────────────────────────────────────────────────────────────
class ClassProb(BaseModel):
    code: str; name: str; probability: float; malignant: bool

class PredictResponse(BaseModel):
    success: bool
    top_prediction: str
    top_class_name: str
    top_confidence: float
    malignancy_risk: float
    risk_level: str
    all_probabilities: list[ClassProb]
    inference_ms: float
    demo_mode: bool

# ── Demo prediction (when model not trained yet) ──────────────────────────────
def demo_predict():
    import random
    raw = [(c, class_meta["class_names"].get(c,c), random.uniform(5,60), c in class_meta["malignant_classes"])
           for c in class_meta["classes"]]
    total = sum(p for _,_,p,_ in raw)
    ranked = sorted([(c,n,round(p/total*100,2),m) for c,n,p,m in raw], key=lambda x:-x[2])
    top = ranked[0]
    mal = sum(p for _,_,p,m in ranked if m)
    return dict(
        top_prediction=top[0], top_class_name=top[1],
        top_confidence=top[2], malignancy_risk=round(mal,2),
        risk_level="high" if mal>65 else "medium" if mal>30 else "low",
        all_probabilities=[{"code":c,"name":n,"probability":p,"malignant":m} for c,n,p,m in ranked],
        demo_mode=True
    )

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status":"ok","device":str(DEVICE),"model_loaded":model is not None,"demo_mode":model is None}

@app.get("/classes")
async def classes():
    return class_meta

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg","image/png","image/webp","image/jpg"):
        raise HTTPException(400, f"Unsupported type: {file.content_type}")

    # Demo mode
    if model is None:
        return PredictResponse(success=True, inference_ms=8.0, **demo_predict())

    # Read + preprocess
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Cannot read image: {e}")

    t0 = time.perf_counter()
    try:
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    ms = (time.perf_counter() - t0) * 1000

    idx_to_class = class_meta.get("idx_to_class") or {i:c for i,c in enumerate(class_meta["classes"])}
    malignant    = set(class_meta["malignant_classes"])

    all_probs = sorted([{
        "code":        idx_to_class.get(i, str(i)),
        "name":        class_meta["class_names"].get(idx_to_class.get(i,""), "Unknown"),
        "probability": round(float(p)*100, 2),
        "malignant":   idx_to_class.get(i,"") in malignant,
    } for i, p in enumerate(probs)], key=lambda x:-x["probability"])

    top = all_probs[0]
    mal = sum(p["probability"] for p in all_probs if p["malignant"])

    return PredictResponse(
        success=True,
        top_prediction=top["code"], top_class_name=top["name"],
        top_confidence=top["probability"], malignancy_risk=round(mal,2),
        risk_level="high" if mal>65 else "medium" if mal>30 else "low",
        all_probabilities=all_probs, inference_ms=round(ms,2), demo_mode=False
    )
