# app.py
import io, os, json, base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

from model_def import GC_MuPeN

ARTIFACT_DIR = "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

CFG = _load_json(os.path.join(ARTIFACT_DIR, "config.json"))
TCFG = _load_json(os.path.join(ARTIFACT_DIR, "temperature.json"))

IMG_SIZE = int(CFG["image_size"])
TEMPERATURE = float(TCFG["temperature"])

DRUG_GROUPS = CFG["drug_groups"]
GUIDELINE_STEPS = CFG["guideline_steps"]

LAB_FEATURES = CFG["lab_features"]
CLIN_FEATURES = CFG["clinical_features"]

LAB_MEANS = np.array(CFG["lab_means"], dtype=np.float32)
LAB_STDS  = np.array(CFG["lab_stds"], dtype=np.float32)
CLIN_MEANS = np.array(CFG["clinical_means"], dtype=np.float32)
CLIN_STDS  = np.array(CFG["clinical_stds"], dtype=np.float32)


# -------------------------
# Load OOD stats
# -------------------------
OOD_PATH = os.path.join(ARTIFACT_DIR, "ood_stats.json")
OOD = _load_json(OOD_PATH) if os.path.exists(OOD_PATH) else None
OOD_MU = np.array(OOD["mu"], dtype=np.float32) if OOD else None
OOD_SD = np.array(OOD["sd"], dtype=np.float32) if OOD else None
OOD_Z_THR = float(OOD["z_thr"]) if OOD else None


# -------------------------
# Load Prototypes
# - supports old key: dist_threshold
# - supports new key: dist_threshold_cosine (recommended)
# -------------------------
PROTO_PATH = os.path.join(ARTIFACT_DIR, "prototypes.json")
PROTO = _load_json(PROTO_PATH) if os.path.exists(PROTO_PATH) else None

# Prefer cosine threshold if present
PROTO_THR = None
if PROTO:
    if "dist_threshold_cosine" in PROTO:
        PROTO_THR = float(PROTO["dist_threshold_cosine"])
    else:
        # backward compatibility
        PROTO_THR = float(PROTO.get("dist_threshold", 0.35))

PROTO_MU = {k: np.array(v["mu"], dtype=np.float32) for k, v in (PROTO["prototypes"].items() if PROTO else [])}


# -------------------------
# Model
# -------------------------
model = GC_MuPeN(
    lab_in_dim=len(LAB_FEATURES),
    clinical_in_dim=len(CLIN_FEATURES),
    num_drugs=len(DRUG_GROUPS),
    num_steps=len(GUIDELINE_STEPS),
).to(DEVICE)

state = torch.load(os.path.join(ARTIFACT_DIR, "model.pt"), map_location=DEVICE)
model.load_state_dict(state, strict=True)
model.eval()

# -------------------------
# Preprocess
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def _decode_image(b: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

def _norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-8)

# -------------------------
# Safety logic
# -------------------------
def violation_score_from_conf(conf: float) -> float:
    v = max(0.0, 0.52 - conf) * 0.1
    return float(min(1.0, v))

def decide(conf: float, vscore: float):
    adjusted = vscore >= 0.08
    abstain = vscore >= 0.18
    if abstain:
        return "ABSTAIN", adjusted, True
    if adjusted:
        return "REVIEW", True, False
    return "RECOMMEND", False, False

# -------------------------
# Grad-CAM helpers (FIXED)
# -------------------------
def _to_base64_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _make_overlay(original: Image.Image, cam: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """
    Robust overlay:
    - original forced RGB
    - cam can be [H,W] or [1,H,W] etc
    - cam is resized to IMG_SIZE x IMG_SIZE before blending
    """
    img = original.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img).astype(np.float32)  # [H,W,3]

    cam = np.array(cam)
    cam = np.squeeze(cam)  # make it [h,w] e.g. 7x7

    # normalize cam
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # ✅ RESIZE CAM to match image size
    cam_img = Image.fromarray((cam * 255.0).astype(np.uint8))
    cam_img = cam_img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC)
    heat = np.array(cam_img).astype(np.float32)  # [IMG_SIZE,IMG_SIZE]

    overlay = img_np.copy()
    overlay[..., 0] = np.clip((1 - alpha) * overlay[..., 0] + alpha * heat, 0, 255)

    return Image.fromarray(overlay.astype(np.uint8))


def compute_gradcam_overlay_base64(img_pil: Image.Image, img_t: torch.Tensor, lab_t: torch.Tensor, clin_t: torch.Tensor, class_idx: int):
    """
    Uses model.forward_with_maps() and feature_map.retain_grad() (in model_def.py).
    Returns base64 PNG overlay or None.
    """
    img_t = img_t.clone().detach().requires_grad_(True)

    drug_logits, step_logits, gate_w3, film_stats, feat_map = model.forward_with_maps(img_t, lab_t, clin_t)

    drug_logits = drug_logits / max(1e-8, TEMPERATURE)
    score = drug_logits[0, class_idx]

    model.zero_grad(set_to_none=True)
    score.backward()

    grads = feat_map.grad  # [B,C,H,W]
    if grads is None:
        return None

    # Grad-CAM
    weights = grads.mean(dim=(2, 3), keepdim=True)           # [1,C,1,1]
    cam = (weights * feat_map).sum(dim=1, keepdim=False)     # [1,H,W]
    cam = F.relu(cam)
    cam_np = cam.detach().cpu().numpy()[0]                   # [H,W]

    overlay = _make_overlay(img_pil, cam_np)
    return _to_base64_png(overlay)



# -------------------------
# OOD helper
# -------------------------
def _ood_z(pooled_feat: np.ndarray) -> float:
    z = np.mean(np.abs((pooled_feat - OOD_MU) / (OOD_SD + 1e-8)))
    return float(z)


def is_valid_skin_image(img_t: torch.Tensor):
    if OOD is None:
        return True, 0.0

    with torch.no_grad():
        pooled, _ = model._image_features_and_map(img_t)
        pooled_np = pooled.squeeze(0).detach().cpu().numpy()

    z = _ood_z(pooled_np)
    return (z <= OOD_Z_THR), z


# -------------------------
# Prototype helper (COSINE DISTANCE)
# -------------------------
def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-8)


def urticaria_likeness(img_t: torch.Tensor):
    if PROTO is None or PROTO_THR is None or len(PROTO_MU) == 0:
        return True, 0.0, None  # skip gate

    with torch.no_grad():
        pooled, _ = model._image_features_and_map(img_t)
        feat = pooled.squeeze(0).detach().cpu().numpy().astype(np.float32)

    feat_n = _l2norm(feat)

    best_g, best_d = None, 1e9
    for g, mu in PROTO_MU.items():
        mu_n = _l2norm(mu.astype(np.float32))
        cos_sim = float(np.dot(feat_n, mu_n))      # [-1, 1]
        d = 1.0 - cos_sim                          # [0, 2] (0=similar)
        if d < best_d:
            best_d, best_g = d, g

    return (best_d <= PROTO_THR), best_d, best_g

# -------------------------
# API
# -------------------------
app = FastAPI(title="AURA GC-MuPeN v3 Safe Backend", version="1.1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "image_size": IMG_SIZE,
        "temperature": TEMPERATURE,
        "lab_features": LAB_FEATURES,
        "clinical_features": CLIN_FEATURES,

        "ood_loaded": bool(OOD is not None),
        "ood_threshold": OOD_Z_THR,
        "ood_path": OOD_PATH,

        "proto_loaded": bool(PROTO is not None),
        "proto_threshold": PROTO_THR,
        "proto_path": PROTO_PATH,
        "proto_num_classes": (len(PROTO_MU) if PROTO else 0),
        "proto_metric": ("cosine" if (PROTO and "dist_threshold_cosine" in PROTO) else "legacy"),
    }

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),

    # ---- LABS (exact order from config.json) ----
    CRP: float = Form(...),
    FT4: float = Form(...),
    IgE: float = Form(...),
    VitD: float = Form(...),
    Age: float = Form(...),

    # ---- CLINICAL (exact order from config.json) ----
    Weight: float = Form(...),
    Height: float = Form(...),
    AgeFirstSymptoms: float = Form(...),
    DiagnosedAge: float = Form(...),
    ItchingScore: float = Form(...),
    AngioedemaDrugQ: float = Form(...),

    # ---- EXPLAINABILITY ----
    return_gradcam: int = Form(0),  # 1 to include base64 overlay
):
    img_pil = _decode_image(await image.read())
    img_t = transform(img_pil).unsqueeze(0).to(DEVICE)

    # 1) OOD gate (skin-domain)
    valid_img, ood_z = is_valid_skin_image(img_t)
    if not valid_img:
        return {
            "final_decision": "INVALID_IMAGE_DOMAIN",
            "invalid_image": True,
            "abstain": False,
            "abstain_reason": "The uploaded image appears to be out-of-distribution (invalid).",
            "image_domain_check": {"passed": False, "ood_z": ood_z, "threshold": OOD_Z_THR},
        }

    # 2) Prototype gate (urticaria-likeness)
    ok_like, proto_d, nearest = urticaria_likeness(img_t)
    if not ok_like:
        return {
            "final_decision": "INVALID_IMAGE_DOMAIN",
            "abstain": True,
            "abstain_reason": "The uploaded image appears to be out-of-distribution (invalid).",
            "prototype_check": {"passed": False, "nearest_class": nearest, "dist": proto_d, "threshold": PROTO_THR, "metric": "cosine"},
        }

    # 3) build lab/clinical tensors
    lab_raw = np.array([CRP, FT4, IgE, VitD, Age], dtype=np.float32)
    lab_vec = _norm(lab_raw, LAB_MEANS, LAB_STDS)
    lab_t = torch.tensor(lab_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    clin_raw = np.array([Weight, Height, AgeFirstSymptoms, DiagnosedAge, ItchingScore, AngioedemaDrugQ], dtype=np.float32)
    clin_vec = _norm(clin_raw, CLIN_MEANS, CLIN_STDS)
    clin_t = torch.tensor(clin_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    if int(return_gradcam) == 1:
        # run with grads to generate Grad-CAM
        drug_logits, step_logits, gate_w3, film_stats = model(img_t, lab_t, clin_t)
        drug_logits = drug_logits / max(1e-8, TEMPERATURE)
        probs = F.softmax(drug_logits, dim=-1).squeeze(0).detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

        gradcam_b64 = compute_gradcam_overlay_base64(img_pil, img_t, lab_t, clin_t, pred_idx)
    else:
        with torch.no_grad():
            drug_logits, step_logits, gate_w3, film_stats = model(img_t, lab_t, clin_t)
            drug_logits = drug_logits / max(1e-8, TEMPERATURE)
            probs = F.softmax(drug_logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        gradcam_b64 = None

    pred_drug = DRUG_GROUPS[pred_idx]
    step_idx = int(torch.argmax(step_logits, dim=-1).item())
    pred_step = GUIDELINE_STEPS[step_idx] if step_idx < len(GUIDELINE_STEPS) else f"STEP_{step_idx+1}"

    # Safety layer
    vscore = violation_score_from_conf(conf)
    final_decision, adjusted_by_filter, abstain = decide(conf, vscore)

    # Modality weights (stabilized and converted to percentages)
    g3 = gate_w3.squeeze(0).detach().cpu().numpy()
    eps = 1e-6
    g3 = np.clip(g3, eps, 1.0)
    g3 = g3 / g3.sum()
    
    modality_weights = [float(g3[0] * 100), float(g3[1] * 100), float(g3[2] * 100)]
    
    # Convert FiLM contribution to percentages (sum to 100%)
    if film_stats and film_stats.get("enabled"):
        gamma_l2 = film_stats.get("gamma_l2_mean", 0.0)
        beta_l2 = film_stats.get("beta_l2_mean", 0.0)
        delta_l2 = film_stats.get("image_delta_l2_mean", 0.0)
        
        total = gamma_l2 + beta_l2 + delta_l2
        if total > 0:
            film_stats["gamma_l2_mean"] = float((gamma_l2 / total) * 100)
            film_stats["beta_l2_mean"] = float((beta_l2 / total) * 100)
            film_stats["image_delta_l2_mean"] = float((delta_l2 / total) * 100)

    return {
        "final_decision": final_decision,
        "pred_drug": pred_drug,
        "pred_step": pred_step,
        "confidence": conf,
        "violation_score": float(vscore),
        "adjusted_by_filter": bool(adjusted_by_filter),
        "abstain": bool(abstain),

        # explainability
        "modality_weights": modality_weights,
        "film_contribution": film_stats,
        "gradcam_overlay_png_base64": gradcam_b64,

        # gates (optional but useful for UI/debug)
        "image_domain_check": {"passed": True, "ood_z": ood_z, "threshold": OOD_Z_THR},
        "prototype_check": {"passed": True, "nearest_class": nearest, "dist": proto_d, "threshold": PROTO_THR, "metric": "cosine"},
    }
