import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class GatedFusionMTL(nn.Module):
    """
    Must match the training architecture exactly.
    Outputs:
      - logits_type: [B,3]
      - logits_risk: [B,2]
      - logits_side: [B,3]
      - mu: [B]
      - lv: [B]  (log variance)
      - gates: dict of mean gate values (floats)
    """

    def __init__(self, tab_dim: int, text_model_name: str, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size

        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        def gate_net():
            return nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.Sigmoid(),
            )

        self.gate_type = gate_net()
        self.gate_risk = gate_net()
        self.gate_side = gate_net()
        self.gate_sev = gate_net()

        self.head_type = nn.Linear(hidden, 3)
        self.head_risk = nn.Linear(hidden, 2)
        self.head_side = nn.Linear(hidden, 3)

        self.head_sev_mu = nn.Linear(hidden, 1)
        self.head_sev_lv = nn.Linear(hidden, 1)  # log variance

    def _fuse(self, z_tab: torch.Tensor, z_txt: torch.Tensor, gate: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        g = gate(torch.cat([z_tab, z_txt], dim=1))
        h = g * z_tab + (1.0 - g) * z_txt
        return h, g

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, tab: torch.Tensor):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # CLS

        z_tab = self.tab_proj(tab)
        z_txt = self.text_proj(cls)

        h_type, g_type = self._fuse(z_tab, z_txt, self.gate_type)
        h_risk, g_risk = self._fuse(z_tab, z_txt, self.gate_risk)
        h_side, g_side = self._fuse(z_tab, z_txt, self.gate_side)
        h_sev, g_sev = self._fuse(z_tab, z_txt, self.gate_sev)

        logits_type = self.head_type(h_type)
        logits_risk = self.head_risk(h_risk)
        logits_side = self.head_side(h_side)

        mu = self.head_sev_mu(h_sev).squeeze(1)
        lv = self.head_sev_lv(h_sev).squeeze(1)

        gates = {
            "type": float(g_type.mean().detach().cpu().item()),
            "risk": float(g_risk.mean().detach().cpu().item()),
            "side": float(g_side.mean().detach().cpu().item()),
            "sev": float(g_sev.mean().detach().cpu().item()),
        }

        return logits_type, logits_risk, logits_side, mu, lv, gates


@dataclass
class LoadedArtifacts:
    config: Dict[str, Any]
    preprocess: Any
    tokenizer: Any
    model: GatedFusionMTL
    device: torch.device


class Predictor:
    """
    Loads artifacts and runs inference on one sample at a time.
    Adds:
      - Confidence-aware abstention (clinical decision safety)
      - Severity clipping (prevents invalid negative CI)
      - Safety audit (rules_triggered + violation_score)
    """

    def __init__(
        self,
        config_path: str,
        preprocess_path: str,
        weights_path: str,
        device: Optional[str] = None,
    ):
        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # config
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        # preprocess (ColumnTransformer)
        self.preprocess = joblib.load(preprocess_path)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["text_model"])

        # model init (tab_dim must match preprocess output dimension)
        tab_dim = self._infer_tab_dim()
        self.model = GatedFusionMTL(
            tab_dim=tab_dim,
            text_model_name=self.cfg["text_model"],
            hidden=256,
            dropout=0.2,
        ).to(self.device)

        # load weights
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # caches
        self.type_classes: List[str] = self.cfg["type_classes"]
        self.side_classes: List[str] = self.cfg["sidefx_classes"]
        self.risk_labels: List[str] = self.cfg["risk_labels"]
        self.max_len: int = int(self.cfg.get("max_len", 256))

        self.num_cols: List[str] = list(self.cfg["num_cols"])
        self.cat_cols: List[str] = list(self.cfg["cat_cols"])
        self.note_text_cols: List[str] = list(self.cfg.get("note_text_cols", ["symptoms_raw", "investigations_raw"]))

        # ---- safety thresholds (can be overridden in config.json) ----
        self.type_conf_thresh: float = float(self.cfg.get("type_conf_thresh", 0.55))
        # 3-class max entropy = ln(3) ~= 1.098; use slightly below to detect "nearly uniform"
        self.type_entropy_thresh: float = float(self.cfg.get("type_entropy_thresh", 1.05))

        # severity output range (set to your itching score scale)
        self.sev_min: float = float(self.cfg.get("sev_min", 0.0))
        self.sev_max: float = float(self.cfg.get("sev_max", 10.0))

        # safety rule thresholds
        self.ige_high_thresh: float = float(self.cfg.get("ige_high_thresh", 180.0))
        self.low_side_prob_thresh: float = float(self.cfg.get("low_side_prob_thresh", 0.70))
        self.angio_high_prob_min: float = float(self.cfg.get("angio_high_prob_min", 0.20))

    def _infer_tab_dim(self) -> int:
        dummy = {}
        for c in self.cfg["num_cols"]:
            dummy[c] = 0.0
        for c in self.cfg["cat_cols"]:
            dummy[c] = "Unknown"
        X = pd.DataFrame([dummy])
        Xt = self.preprocess.transform(X)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        return int(Xt.shape[1])

    def _build_tab_df(self, features: Dict[str, Any]) -> pd.DataFrame:
        row: Dict[str, Any] = {}

        # numeric
        for c in self.num_cols:
            v = features.get(c, None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                row[c] = np.nan
            else:
                try:
                    row[c] = float(v)
                except Exception:
                    row[c] = np.nan

        # categorical (auto-fill missing as "Unknown")
        for c in self.cat_cols:
            v = features.get(c, "Unknown")
            row[c] = "Unknown" if v is None else str(v)

        df = pd.DataFrame([row])

        # numeric fallback fill
        for c in self.num_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(0.0)

        return df

    def _build_note_text(self, features: Dict[str, Any]) -> str:
        parts = []
        for c in self.note_text_cols:
            t = features.get(c, "")
            if t is None:
                t = ""
            parts.append(str(t))
        return " [SEP] ".join(parts).strip()

    @torch.no_grad()
    def predict_one(self, features: Dict[str, Any]) -> Dict[str, Any]:
        # ---------- tabular ----------
        tab_df = self._build_tab_df(features)
        tab = self.preprocess.transform(tab_df)
        tab = tab.toarray() if hasattr(tab, "toarray") else np.asarray(tab)
        tab = torch.tensor(tab, dtype=torch.float32, device=self.device)

        # ---------- text ----------
        note = self._build_note_text(features)
        enc = self.tokenizer(
            note,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # ---------- forward ----------
        logits_type, logits_risk, logits_side, mu, lv, gates = self.model(input_ids, attention_mask, tab)

        # ---------- probabilities ----------
        p_type = torch.softmax(logits_type, dim=1).squeeze(0).detach().cpu().numpy()
        p_side = torch.softmax(logits_side, dim=1).squeeze(0).detach().cpu().numpy()
        p_risk = torch.sigmoid(logits_risk).squeeze(0).detach().cpu().numpy()

        # ---------- severity uncertainty (Gaussian) ----------
        mu_val = float(mu.squeeze(0).detach().cpu().item())
        var_val = float(torch.exp(lv.squeeze(0)).detach().cpu().item())
        std_val = math.sqrt(max(var_val, 1e-9))
        ci95 = [mu_val - 1.96 * std_val, mu_val + 1.96 * std_val]

        #  clinical safety: clip severity and CI to valid range
        mu_val = float(np.clip(mu_val, self.sev_min, self.sev_max))
        ci95 = [
            float(np.clip(ci95[0], self.sev_min, self.sev_max)),
            float(np.clip(ci95[1], self.sev_min, self.sev_max)),
        ]

        # ---------- predicted classes ----------
        type_idx = int(np.argmax(p_type))
        side_idx = int(np.argmax(p_side))
        raw_type_pred = self.type_classes[type_idx]
        side_pred = self.side_classes[side_idx]

        # ==========================================================
        #  NOVELTY A: Confidence-aware abstention (decision safety)
        # ==========================================================
        type_conf = float(np.max(p_type))
        type_entropy = float(-np.sum(p_type * np.log(p_type + 1e-12)))

        abstain = (type_conf < self.type_conf_thresh)

        final_decision = "PREDICT"
        final_type_pred = raw_type_pred
        abstain_reason = None

        if abstain:
            final_decision = "ABSTAIN"
            final_type_pred = "ABSTAIN"
            abstain_reason = "Low confidence / high uncertainty in urticaria type"

        # ==========================================================
        #  NOVELTY B: Safety audit layer (rules + violation score)
        # ==========================================================
        rules_triggered: List[str] = []
        violation_score: float = 0.0

        # Rule 1: High IgE but model strongly predicts LOW side risk
        try:
            ige_val = float(features.get("IgE", 0.0))
        except Exception:
            ige_val = 0.0

        low_side_prob = float(p_side[0])  # LOW prob (assuming classes [LOW, MED, HIGH])
        if ige_val >= self.ige_high_thresh and low_side_prob >= self.low_side_prob_thresh:
            rules_triggered.append("HIGH_IGE_BUT_LOW_SIDE_RISK")
            violation_score += (low_side_prob - self.low_side_prob_thresh)

        # Rule 2 (optional): Angioedema present but HIGH risk prob is too low
        ang = str(features.get("If  angioedema is present", "")).strip().lower()
        high_side_prob = float(p_side[2])  # HIGH prob
        if ("yes" in ang) and (high_side_prob < self.angio_high_prob_min):
            rules_triggered.append("ANGIOEDEMA_BUT_LOW_HIGH_RISK")
            violation_score += (self.angio_high_prob_min - high_side_prob)

        safety_flag = (len(rules_triggered) > 0)

        # ---------- response ----------
        return {
            "final_decision": final_decision,
            "abstain": bool(abstain),
            "abstain_reason": abstain_reason,
            "confidence": {
                "type_confidence": type_conf,
                "type_entropy": type_entropy,
                "thresholds": {
                    "type_conf_thresh": float(self.type_conf_thresh),
                    "type_entropy_thresh": float(self.type_entropy_thresh),
                },
            },
            "safety": {
                "safety_flag": bool(safety_flag),
                "rules_triggered": rules_triggered,
                "violation_score": float(violation_score),
            },
            "urticaria_type": {
                "pred": final_type_pred,
                "raw_pred": raw_type_pred,
                "probs": {cls: float(p_type[i]) for i, cls in enumerate(self.type_classes)},
            },
            "side_effect_risk_proxy": {
                "pred": side_pred,
                "probs": {cls: float(p_side[i]) for i, cls in enumerate(self.side_classes)},
            },
            "secondary_risk": {lbl: float(p_risk[i]) for i, lbl in enumerate(self.risk_labels)},
            "severity": {
                "mu": mu_val,
                "var": var_val,
                "ci95": [float(ci95[0]), float(ci95[1])],
                "range": [float(self.sev_min), float(self.sev_max)],
            },
            "fusion_gates_mean": gates,
            "disclaimer": "Research decision-support only. Not medical advice.",
        }
