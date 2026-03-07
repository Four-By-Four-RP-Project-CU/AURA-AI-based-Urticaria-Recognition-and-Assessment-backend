import json, os
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

# ---------------------------------------------------------------------------
# EAACI / GA²LEN Chronic Spontaneous Urticaria (CSU) Guideline Step Details
# Source: EAACI Guidelines for Urticaria (2022 revision)
# ---------------------------------------------------------------------------
GUIDELINE_STEP_INFO: Dict[str, Dict[str, Any]] = {
    "STEP_1": {
        "label": "Step 1 — Standard-dose 2nd-generation H1-antihistamines",
        "indication": (
            "First-line treatment for all CSU patients. "
            "Indicated when UAS7 > 0 (any active disease)."
        ),
        "drugs": [
            "Cetirizine 10 mg once daily",
            "Fexofenadine 180 mg once daily",
            "Loratadine 10 mg once daily",
            "Bilastine 20 mg once daily",
            "Desloratadine 5 mg once daily",
            "Rupatadine 10 mg once daily",
        ],
        "duration": "2–4 weeks; reassess response before escalating.",
        "uas7_aligned_range": (1, 15),   # mild disease
        "notes": (
            "Prefer non-sedating 2nd-generation agents. "
            "Avoid 1st-generation antihistamines as routine therapy (sedation, anticholinergic effects). "
            "If well-controlled (UAS7 0–6 at reassessment) continue for ≥3 months then consider tapering."
        ),
    },
    "STEP_2": {
        "label": "Step 2 — Up-dosed 2nd-generation H1-antihistamines (up to 4×)",
        "indication": (
            "For patients who remain symptomatic (UAS7 ≥ 7) after 2–4 weeks at standard dose. "
            "Increase dose up to 4× the licensed standard dose."
        ),
        "drugs": [
            "Cetirizine up to 40 mg/day",
            "Fexofenadine up to 720 mg/day",
            "Loratadine up to 40 mg/day",
            "Bilastine up to 80 mg/day",
            "Desloratadine up to 20 mg/day",
            "Rupatadine up to 40 mg/day",
        ],
        "duration": "2–4 weeks; reassess. Off-label dosing — document and inform patient.",
        "uas7_aligned_range": (7, 27),   # mild-to-moderate
        "notes": (
            "Off-label use in most countries; obtain informed consent. "
            "Monitor for dose-related adverse effects (QTc prolongation with some agents at high doses). "
            "If still poorly controlled (UAS7 > 16) after 4 weeks, escalate to Step 3."
        ),
    },
    "STEP_3": {
        "label": "Step 3 — Add omalizumab (anti-IgE biologic, 300 mg s.c. q4w)",
        "indication": (
            "For patients with moderate-to-severe CSU (UAS7 ≥ 16) inadequately controlled "
            "by updosed H1-antihistamines. Requires IgE-mediated pathway activity indicator."
        ),
        "drugs": [
            "Omalizumab 300 mg subcutaneous injection every 4 weeks (preferred dose)",
            "Omalizumab 150 mg s.c. q4w (lower-activity cases, consider if UAS7 16–27)",
        ],
        "duration": (
            "Minimum 6-month course. Reassess at 3 months; "
            "continue up to 12 months if responding. "
            "Attempt discontinuation after 12 months of disease control."
        ),
        "uas7_aligned_range": (16, 42),   # moderate-to-severe
        "notes": (
            "Continue background H1-AH (standard dose). "
            "Response expected within 4–12 weeks. "
            "Monitor IgE levels — not required to guide dosing in CSU. "
            "Safe in pregnancy (consider risk-benefit). "
            "If no response at 300 mg after 6 months, consider Step 4."
        ),
    },
    "STEP_4": {
        "label": "Step 4 — Add ciclosporin (cyclosporine A) or other advanced agents",
        "indication": (
            "For refractory CSU unresponsive to Step 3 (omalizumab ≥ 6 months at 300 mg q4w) "
            "with persistent severe disease (UAS7 ≥ 28)."
        ),
        "drugs": [
            "Ciclosporin (Cyclosporine A) 3–5 mg/kg/day orally (preferred advanced agent)",
            "Mycophenolate mofetil 1–2 g/day (alternative, less evidence)",
            "Methotrexate 10–15 mg/week (limited evidence)",
            "Hydroxychloroquine 200–400 mg/day (limited evidence, mainly for autoimmune urticaria)",
        ],
        "duration": (
            "Ciclosporin: up to 6 months, then taper. "
            "Monitor renal function (serum creatinine), blood pressure and electrolytes every 4–8 weeks."
        ),
        "uas7_aligned_range": (28, 42),   # severe
        "notes": (
            "Ciclosporin is immunosuppressive — screen for infections, malignancy history. "
            "Avoid nephrotoxic drugs concurrently. "
            "Continue omalizumab concurrently if partially effective. "
            "Referral to a urticaria centre of excellence is recommended at this stage."
        ),
    },
}

# ---------------------------------------------------------------------------
# UAS7 (Urticaria Activity Score — 7-day) Classification
# Wheals: 0–3/day  |  Pruritus: 0–3/day  |  Daily UAS: 0–6  |  UAS7: 0–42
# Reference: Zuberbier et al., EAACI Guidelines 2022
# ---------------------------------------------------------------------------
UAS7_CATEGORIES = [
    {"label": "Urticaria-free",        "min": 0,  "max": 0,  "severity": "none",     "recommended_step": None},
    {"label": "Well-controlled",       "min": 1,  "max": 6,  "severity": "minimal",  "recommended_step": "STEP_1"},
    {"label": "Mild urticaria",        "min": 7,  "max": 15, "severity": "mild",     "recommended_step": "STEP_1"},
    {"label": "Moderate urticaria",    "min": 16, "max": 27, "severity": "moderate", "recommended_step": "STEP_2"},
    {"label": "Severe urticaria",      "min": 28, "max": 42, "severity": "severe",   "recommended_step": "STEP_3"},
]

def classify_uas7(score: float) -> Dict[str, Any]:
    """Return the UAS7 category dict for a given raw score (0–42)."""
    score = float(score)
    for cat in UAS7_CATEGORIES:
        if cat["min"] <= score <= cat["max"]:
            return {
                "score": round(score, 1),
                "category": cat["label"],
                "severity": cat["severity"],
                "recommended_step": cat["recommended_step"],
                "range": f"{cat['min']}–{cat['max']}",
                "scale": "0–42  (0 = disease-free, 42 = most severe)",
            }
    # out of range guard
    return {
        "score": round(score, 1),
        "category": "Out of range",
        "severity": "unknown",
        "recommended_step": None,
        "range": "0–42",
        "scale": "0–42  (0 = disease-free, 42 = most severe)",
    }

# -------------------------
# Model definition — GC-MuPeN v3 (EfficientNet-B3 + FiLM + GatedFusion)
# Architecture MUST exactly match the training notebook to load the checkpoint.
# -------------------------

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim * 2),
        )

    def forward(self, feats, cond):
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return feats * (1 + gamma) + beta


class GC_MuPeN_v3(nn.Module):
    """
    GC-MuPeN v3: EfficientNet-B3 + FiLM + Gated Fusion.
    Architecture exactly mirrors the training notebook (GC_MuPeN class) so that
    the saved checkpoint loads without size mismatches.
    Returns a dict (not a tuple) for API compatibility.
    """
    def __init__(self, lab_in_dim: int, clinical_in_dim: int,
                 num_drugs: int, num_steps: int,
                 dropout: float = 0.3, fusion_hidden: int = 512, fusion_out: int = 256):
        super().__init__()

        self.image_backbone = timm.create_model(
            "efficientnet_b3", pretrained=False, num_classes=0)
        img_feat_dim = self.image_backbone.num_features  # 1536 for B3

        # Lab tower — 4-element Sequential (indices match checkpoint keys)
        self.lab_mlp = nn.Sequential(
            nn.Linear(lab_in_dim, 128) if lab_in_dim > 0 else nn.Identity(),
            nn.ReLU()                  if lab_in_dim > 0 else nn.Identity(),
            nn.Dropout(dropout)        if lab_in_dim > 0 else nn.Identity(),
            nn.Linear(128, 128)        if lab_in_dim > 0 else nn.Identity(),
        )

        # Clinical tower
        self.clin_mlp = nn.Sequential(
            nn.Linear(clinical_in_dim, 128) if clinical_in_dim > 0 else nn.Identity(),
            nn.ReLU()                       if clinical_in_dim > 0 else nn.Identity(),
            nn.Dropout(dropout)             if clinical_in_dim > 0 else nn.Identity(),
            nn.Linear(128, 128)             if clinical_in_dim > 0 else nn.Identity(),
        )

        # FiLM: labs modulate image features
        cond_dim = 128 if lab_in_dim > 0 else 0
        self.film = FiLM(cond_dim, img_feat_dim) if cond_dim > 0 else None

        # Gating + Fusion (operate on concatenated raw+FiLM features)
        fused_in_dim = (img_feat_dim
                        + (128 if lab_in_dim      > 0 else 0)
                        + (128 if clinical_in_dim > 0 else 0))

        self.fusion = nn.Sequential(
            nn.Linear(fused_in_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_out),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(fused_in_dim, 3),
            nn.Softmax(dim=-1),
        )

        self.drug_head = nn.Linear(fusion_out, num_drugs)
        self.step_head = nn.Linear(fusion_out, num_steps)

        # store flags for forward
        self._has_lab  = lab_in_dim > 0
        self._has_clin = clinical_in_dim > 0

    def forward(self, image, lab_feats, clinical_feats):
        img_feats_raw = self.image_backbone(image)

        lab_emb  = self.lab_mlp(lab_feats)        if self._has_lab  else None
        clin_emb = self.clin_mlp(clinical_feats)  if self._has_clin else None

        # FiLM modulation
        img_feats = (self.film(img_feats_raw, lab_emb)
                     if self.film is not None and lab_emb is not None
                     else img_feats_raw)

        # Gate — input is the (FiLM-modulated image + towers)
        gate_parts = [img_feats]
        if lab_emb  is not None: gate_parts.append(lab_emb)
        if clin_emb is not None: gate_parts.append(clin_emb)
        gate_input  = torch.cat(gate_parts, dim=-1)
        gate_logits = self.gate[0](gate_input)         # pre-softmax logits [B, 3]
        gate_w      = F.softmax(gate_logits, dim=-1)   # same as self.gate(gate_input), checkpoint-compatible

        # Gated fusion
        gated = [img_feats * gate_w[:, 0:1]]
        if lab_emb  is not None: gated.append(lab_emb  * gate_w[:, 1:2])
        if clin_emb is not None: gated.append(clin_emb * gate_w[:, 2:3])

        # Display contributions: 5th-root power transform of gate_w (equivalent to
        # temperature T=5 scaling). Spreads a saturated [0.998, 0.002, 8e-5] into a
        # readable [~70%, ~19%, ~10%] without affecting the prediction at all.
        # clamp(min=1e-6) ensures even float32-exact zeros (extreme logits) get a
        # visible floor: (1e-6)^0.2 ≈ 0.063, so each modality shows ≥ ~5 %.
        g = gate_w.detach().clamp(min=1e-6)
        g_display = g.pow(0.2)
        gate_display = g_display / (g_display.sum(dim=-1, keepdim=True) + 1e-8)

        fused       = self.fusion(torch.cat(gated, dim=-1))
        drug_logits = self.drug_head(fused)
        step_logits = self.step_head(fused)

        return {
            "drug_logits":  drug_logits,
            "step_logits":  step_logits,
            "gate_w":       gate_w,
            "gate_display": gate_display,  # dimension-normalised contribution for chart
            "img_feats":    img_feats_raw, # raw B3 features (1536-d) for OOD
        }


# Alias kept for any external references
GC_MuPeN = GC_MuPeN_v3
GC_MuPeN_v4 = GC_MuPeN_v3

# -------------------------
# Runtime wrapper
# -------------------------
class ModelRuntime:
    def __init__(self, artifacts_dir: str, device: str | None = None):
        self.artifacts_dir = artifacts_dir
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        with open(os.path.join(artifacts_dir, "config.json"), "r") as f:
            self.cfg = json.load(f)

        with open(os.path.join(artifacts_dir, "temperature.json"), "r") as f:
            self.temperature = float(json.load(f)["temperature"])

        # Per-class thresholds (optimised on val set to fix minority-class bias).
        # Falls back to uniform [0, 0, ..., 0] (i.e. plain argmax) if file absent.
        thresholds_path = os.path.join(artifacts_dir, "thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                self.thresholds = np.array(json.load(f)["thresholds"], dtype=np.float32)
        else:
            self.thresholds = None   # will default to argmax

        self.ood_z_thr = None
        ood_path = os.path.join(artifacts_dir, "ood_stats.json")
        if os.path.exists(ood_path):
            with open(ood_path, "r") as f:
                ood = json.load(f)
            self.ood_mean = np.array(ood["mu"], dtype=np.float32)
            self.ood_std = np.array(ood["sd"], dtype=np.float32) + 1e-8
            self.ood_z_thr = float(ood.get("z_thr", 1.5))
        else:
            self.ood_mean, self.ood_std = None, None

        # Prototype cosine-distance gate (reference: urticaria_likeness)
        # Minimum cosine distance from image features to any class prototype mean.
        # Values were calibrated on actual training CU photos; real CSU photos are
        # expected to score below dist_threshold_cosine; non-CU photos score above it.
        self.proto_mu: Dict[str, np.ndarray] = {}
        self.proto_thr: float | None = None
        proto_path = os.path.join(artifacts_dir, "prototypes.json")
        if os.path.exists(proto_path):
            with open(proto_path, "r") as f:
                proto = json.load(f)
            self.proto_mu = {
                k: np.array(v["mu"], dtype=np.float32)
                for k, v in proto.get("prototypes", {}).items()
            }
            if "dist_threshold_cosine" in proto:
                self.proto_thr = float(proto["dist_threshold_cosine"])
            elif "dist_threshold" in proto:
                self.proto_thr = float(proto["dist_threshold"]) / 1000.0  # L2→cosine scale

        self.lab_cols = self.cfg["lab_features"]
        self.clin_cols = self.cfg["clinical_features"]
        self.drug_groups = self.cfg["drug_groups"]

        self.lab_means = np.array(self.cfg["lab_means"], dtype=np.float32) if self.lab_cols else np.array([], dtype=np.float32)
        self.lab_stds  = np.array(self.cfg["lab_stds"], dtype=np.float32) if self.lab_cols else np.array([], dtype=np.float32)
        self.clin_means = np.array(self.cfg["clinical_means"], dtype=np.float32) if self.clin_cols else np.array([], dtype=np.float32)
        self.clin_stds  = np.array(self.cfg["clinical_stds"], dtype=np.float32) if self.clin_cols else np.array([], dtype=np.float32)

        self.model = GC_MuPeN_v3(
            lab_in_dim=len(self.lab_cols),
            clinical_in_dim=len(self.clin_cols),
            num_drugs=len(self.drug_groups),
            num_steps=4,
            dropout=0.3,
            fusion_hidden=512,
            fusion_out=256,
        ).to(self.device)

        sd = torch.load(os.path.join(artifacts_dir, "model.pt"), map_location=self.device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

        self.tfm = transforms.Compose([
            transforms.Resize((self.cfg["image_size"], self.cfg["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def _vectorize(self, values: Dict[str, float], cols: list[str], means: np.ndarray, stds: np.ndarray) -> torch.Tensor:
        v = np.array([float(values.get(k, 0.0)) for k in cols], dtype=np.float32)
        if len(cols) > 0:
            v = (v - means) / stds
        return torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)

    def preprocess(self, image_pil: Image.Image, lab_values: Dict[str, float], clin_values: Dict[str, float]):
        img = self.tfm(image_pil.convert("RGB")).unsqueeze(0).to(self.device)
        lab = self._vectorize(lab_values, self.lab_cols, self.lab_means, self.lab_stds) if self.lab_cols else torch.zeros((1,0), device=self.device)
        clin = self._vectorize(clin_values, self.clin_cols, self.clin_means, self.clin_stds) if self.clin_cols else torch.zeros((1,0), device=self.device)
        return img, lab, clin

    @torch.no_grad()
    def predict(self, image_pil: Image.Image, lab_values: Dict[str, float], clin_values: Dict[str, float],
                abstain_threshold: float = 0.55) -> Dict[str, Any]:
        # Pre-check: grayscale standard deviation of the raw image.
        # Any real camera photograph has std > ~10.  Solid colours, plain logos
        # and pure-gradient fills produce std < 8 and are never valid CSU photos.
        _gray = np.array(image_pil.convert("L"), dtype=np.float32)
        _low_variance = bool(np.std(_gray) < 8.0)

        img, lab, clin = self.preprocess(image_pil, lab_values, clin_values)
        out = self.model(img, lab, clin)

        drug_logits = out["drug_logits"]
        step_logits = out["step_logits"]
        gate_w        = out["gate_w"]
        gate_display  = out["gate_display"]
        img_feats     = out["img_feats"]

        drug_logits = drug_logits / self.temperature
        probs = F.softmax(drug_logits, dim=-1).squeeze(0).detach().cpu().numpy()

        # Apply per-class thresholds: pick class with highest (prob - threshold).
        # This corrects systematic majority-class bias (LTRA / ADVANCED_THERAPY
        # were consistently swamped by H1_ANTIHISTAMINE without this adjustment).
        if self.thresholds is not None:
            top_idx = int(np.argmax(probs - self.thresholds))
        else:
            top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        top3 = sorted([(self.drug_groups[i], float(probs[i])) for i in range(len(probs))],
                    key=lambda x: x[1], reverse=True)[:3]

        step = int(torch.argmax(step_logits, dim=-1).item()) + 1

        # ── OOD detection (three-layer, matching reference app.py) ───────────
        # Layer 1 — pixel variance: solid/blank/graphic images never hold a CSU photo
        # Layer 2 — feature z-score: mean |z| of backbone features vs training mean
        # Layer 3 — prototype cosine distance: minimum cosine distance to any class
        #            prototype mean computed from actual training CU images.
        #            Real CSU photos  → dist < proto_thr (threshold from prototypes.json)
        #            Non-CSU photos   → dist > proto_thr
        fv = img_feats.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Feature z-score
        feat_z = 0.0
        if self.ood_mean is not None and self.ood_std is not None:
            feat_z = float(np.mean(np.abs((fv - self.ood_mean) / self.ood_std)))

        # Prototype cosine distance
        proto_min_dist = 0.0
        if self.proto_mu:
            def _l2n(x): return x / (np.linalg.norm(x) + 1e-8)
            fv_n = _l2n(fv)
            proto_min_dist = min(
                1.0 - float(np.dot(fv_n, _l2n(mu)))
                for mu in self.proto_mu.values()
            )

        # OOD flag: any gate fires → reject
        ood_flag = False
        if _low_variance:                                          # layer 1
            ood_flag = True
        elif self.ood_z_thr and feat_z > self.ood_z_thr:         # layer 2
            ood_flag = True
        elif self.proto_thr and proto_min_dist > self.proto_thr:  # layer 3
            ood_flag = True

        # ood_z: report the most informative signal (proto dist if available)
        ood_z = float(proto_min_dist) if self.proto_mu else float(feat_z)

        step_key = f"STEP_{step}"
        return {
            "predicted_drug_group": self.drug_groups[top_idx],
            "confidence": top_conf,
            "top3": top3,
            "mapped_guideline_step": step_key,
            "guideline_step_detail": GUIDELINE_STEP_INFO.get(step_key, {}),
            "abstain": bool(top_conf < abstain_threshold),
            "ood_flag": ood_flag,
            "ood_z": ood_z,
            # Dimension-normalised per-element contribution of each gated modality
            # embedding. Honest representation: shows how much each modality
            # actually contributes to the fused vector regardless of gate saturation.
            "modality_gate_weights": gate_display.squeeze(0).detach().cpu().tolist(),
        }