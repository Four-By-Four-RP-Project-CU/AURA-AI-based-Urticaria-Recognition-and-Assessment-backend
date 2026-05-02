import json
import os
import logging
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
try:
    from huggingface_hub import snapshot_download as _hf_snapshot_download
except Exception:  # huggingface_hub may be absent in some environments
    _hf_snapshot_download = None


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture  (must match training notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """Learned single-head attention pooling over BERT token sequence."""
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


class GatedFusionMTL(nn.Module):
    """Task-Conditioned Gated Fusion MTL — mirrors ClinicalSafe_CU_MTL_Improved."""

    def __init__(self, tab_dim: int, text_model_name: str, hidden: int = 512, dropout: float = 0.3, hf_token: str | None = None):
        super().__init__()
        hf_kwargs = {"token": hf_token} if hf_token else {}
        self.text_encoder = AutoModel.from_pretrained(text_model_name, **hf_kwargs)
        text_dim = self.text_encoder.config.hidden_size

        self.attn_pool = AttentionPooling(text_dim)

        self.tab_proj = nn.Linear(tab_dim, hidden)
        self.tab_res1 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.tab_res2 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )

        def gate_net():
            return nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),     nn.Sigmoid(),
            )

        self.gate_type = gate_net()
        self.gate_risk = gate_net()
        self.gate_side = gate_net()
        self.gate_sev  = gate_net()

        self.norm_type = nn.LayerNorm(hidden)
        self.norm_risk = nn.LayerNorm(hidden)
        self.norm_side = nn.LayerNorm(hidden)
        self.norm_sev  = nn.LayerNorm(hidden)

        def mlp_head(n_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, n_out),
            )

        self.head_type   = mlp_head(3)
        self.head_risk   = mlp_head(2)
        self.head_side   = mlp_head(3)
        self.head_sev_mu = nn.Linear(hidden, 1)
        self.head_sev_lv = nn.Linear(hidden, 1)

        self.log_var = nn.Parameter(torch.zeros(4))

    def _encode_tab(self, tab: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.tab_proj(tab))
        h = h + self.tab_res1(h)
        h = h + self.tab_res2(h)
        return h

    def fuse(self, z_tab, z_txt, gate_net, norm):
        g = gate_net(torch.cat([z_tab, z_txt], dim=1))
        h = g * z_tab + (1 - g) * z_txt
        return norm(h), g

    def forward(self, input_ids, attention_mask, tab):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.attn_pool(out.last_hidden_state, attention_mask)

        z_tab = self._encode_tab(tab)
        z_txt = self.text_proj(cls)
        z_tab = self.shared(z_tab)
        z_txt = self.shared(z_txt)

        h_type, g_type = self.fuse(z_tab, z_txt, self.gate_type, self.norm_type)
        h_risk, g_risk = self.fuse(z_tab, z_txt, self.gate_risk, self.norm_risk)
        h_side, g_side = self.fuse(z_tab, z_txt, self.gate_side, self.norm_side)
        h_sev,  g_sev  = self.fuse(z_tab, z_txt, self.gate_sev,  self.norm_sev)

        logits_type = self.head_type(h_type)
        logits_risk = self.head_risk(h_risk)
        logits_side = self.head_side(h_side)
        mu = self.head_sev_mu(h_sev).squeeze(1)
        lv = self.head_sev_lv(h_sev).squeeze(1)

        gates = {
            "type": float(g_type.mean().detach().cpu().item()),
            "risk": float(g_risk.mean().detach().cpu().item()),
            "side": float(g_side.mean().detach().cpu().item()),
            "sev":  float(g_sev.mean().detach().cpu().item()),
        }
        return logits_type, logits_risk, logits_side, mu, lv, gates


# ─────────────────────────────────────────────────────────────────────────────
# Clinical interpretation helpers  (mirrors notebook Cell 10)
# ─────────────────────────────────────────────────────────────────────────────

_SEV_BANDS = [
    (0.0,  3.0, "MILD",     "Minimal impact; standard antihistamine management"),
    (3.0,  6.0, "MODERATE", "Meaningful impairment; scheduled follow-up recommended"),
    (6.0,  8.0, "SEVERE",   "Significant burden; consider second-line / biologic therapy"),
    (8.0, 10.1, "EXTREME",  "Debilitating; urgent escalation to specialist required"),
]


def _sev_band(score: float):
    for lo, hi, label, desc in _SEV_BANDS:
        if lo <= score < hi:
            return label, desc
    return _SEV_BANDS[-1][2], _SEV_BANDS[-1][3]


def _composite_interpretation(score: float) -> str:
    if score >= 0.75:
        return "VERY HIGH — Urgent specialist referral; escalate therapy immediately."
    if score >= 0.55:
        return "HIGH      — Close monitoring; consider biologics (e.g. omalizumab)."
    if score >= 0.35:
        return "MODERATE  — Regular review every 4 weeks; optimise current therapy."
    return     "LOW       — Standard management; annual review sufficient."


# ─────────────────────────────────────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────────────────────────────────────

def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


class Runtime:
    def __init__(self, artifacts_dir: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        hf_repo = _env_first(
            "RISK_MODEL_REPO",
            "IT22607232_MODEL_REPO",
            "HUGGINGFACE_MODEL_REPO_RISK",
        )
        hf_token = _env_first("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
        downloaded = False
        if hf_repo and _hf_snapshot_download is not None:
            try:
                repo_dir = _hf_snapshot_download(repo_id=hf_repo, token=hf_token)
                artifacts_dir = repo_dir
                downloaded = True
            except Exception:
                downloaded = False

        with open(f"{artifacts_dir}/config.json", "r") as f:
            self.cfg = json.load(f)

        self.preprocess  = joblib.load(f"{artifacts_dir}/preprocess.joblib")
        tokenizer_kwargs = {"token": hf_token} if hf_token else {}
        self.tokenizer   = AutoTokenizer.from_pretrained(self.cfg["text_model"], **tokenizer_kwargs)

        # Logging: show whether HF token was detected and which artifacts dir is used
        self.logger = logging.getLogger(__name__)
        if hf_repo:
            if downloaded:
                self.logger.info("Downloaded risk artifacts from HF repo '%s' into %s", hf_repo, artifacts_dir)
            else:
                self.logger.info("RISK_MODEL_REPO set to '%s' but download failed; using local artifacts at %s", hf_repo, artifacts_dir)
        if hf_token:
            self.logger.info("HUGGINGFACE_HUB_TOKEN detected — will use it for tokenizer/model loading if needed.")
        else:
            self.logger.info("No HUGGINGFACE_HUB_TOKEN found — loading from local artifacts when available.")
        self.logger.info("Artifacts directory: %s", artifacts_dir)

        # Derive tab_dim dynamically so it always matches the saved preprocess pipeline
        num_cols = self.cfg["num_cols"]
        cat_cols = self.cfg["cat_cols"]
        _dummy   = {c: 0.0 for c in num_cols} | {c: "Unknown" for c in cat_cols}
        _X       = self.preprocess.transform(pd.DataFrame([_dummy])).toarray().astype(np.float32)
        tab_dim  = _X.shape[1]

        arch = self.cfg.get("model_arch", {})
        self.model = GatedFusionMTL(
            tab_dim=tab_dim,
            text_model_name=self.cfg["text_model"],
            hidden=arch.get("hidden", 512),
            dropout=arch.get("dropout", 0.3),
            hf_token=hf_token,
        ).to(self.device)

        state = torch.load(f"{artifacts_dir}/model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.type_classes   = self.cfg["type_classes"]
        self.risk_labels    = self.cfg["risk_labels"]
        self.sidefx_classes = self.cfg["sidefx_classes"]
        self.max_len        = int(self.cfg["max_len"])

        # Post-training calibration scales.
        # Notebook calibration returned [1,1,1] (no correction needed), so the
        # config may not include this key — default to all-ones safely.
        self.type_scales = np.array(
            self.cfg.get("type_scales", [1.0] * len(self.type_classes)),
            dtype=np.float32,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_tab(self, labs: dict, categorical: dict) -> torch.Tensor:
        row = {}
        for c in self.cfg["num_cols"]:
            v = labs.get(c)
            row[c] = float(v) if v is not None else 0.0
        for c in self.cfg["cat_cols"]:
            v = categorical.get(c)
            row[c] = str(v) if (v is not None and str(v).strip() not in ("", "None")) else "Unknown"
        X = self.preprocess.transform(pd.DataFrame([row])).toarray().astype(np.float32)
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def _tokenise(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    # ── Main inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        symptoms_raw: str,
        investigations_raw: str,
        labs: dict,
        categorical: dict,
    ) -> dict:
        """
        Run inference and return a full clinical risk profile dict that mirrors
        the notebook's make_risk_profile() / print_risk_profile() output.

        Parameters
        ----------
        symptoms_raw       : free-text symptom description
        investigations_raw : free-text lab / investigation notes
        labs               : {column_name: float_or_None}  — numeric features
        categorical        : {column_name: str_or_None}   — questionnaire answers
        """
        # Build inputs
        note_text = f"{symptoms_raw or ''} [SEP] {investigations_raw or ''}".strip()
        tab                    = self._build_tab(labs, categorical)
        input_ids, attn_mask   = self._tokenise(note_text)

        # Forward pass
        logits_type, logits_risk, logits_side, mu, lv, gates = self.model(
            input_ids, attn_mask, tab
        )

        # Decode probabilities
        type_probs = torch.softmax(logits_type, dim=1).squeeze(0).cpu().numpy()  # (3,)
        risk_probs = torch.sigmoid(logits_risk).squeeze(0).cpu().numpy()          # (2,)
        side_probs = torch.softmax(logits_side, dim=1).squeeze(0).cpu().numpy()  # (3,)
        sev_mean   = float(mu.squeeze(0).cpu().item())
        sev_std    = float(torch.sqrt(torch.exp(lv.squeeze(0))).cpu().item())

        # Calibrated type prediction (type_scales from post-training grid search)
        cal_probs    = type_probs * self.type_scales
        type_pred_id = int(np.argmax(cal_probs))
        side_pred_id = int(np.argmax(side_probs))

        # Severity: clamp to [0, 10]; 95% CI also clamped
        sev_clamp = float(np.clip(sev_mean, 0.0, 10.0))
        ci_lo     = round(float(max(0.0,  sev_mean - 1.96 * sev_std)), 2)
        ci_hi     = round(float(min(10.0, sev_mean + 1.96 * sev_std)), 2)
        sev_label, sev_desc = _sev_band(sev_clamp)

        # Composite risk score (weights from notebook Cell 10)
        #   35% HIGH side-effect prob + 25% max secondary risk + 25% severity/10 + 15% type confidence
        composite = (
            0.35 * float(side_probs[2]) +
            0.25 * float(risk_probs.max()) +
            0.25 * (sev_clamp / 10.0) +
            0.15 * float(type_probs[type_pred_id])
        )

        return {
            "urticaria_type": {
                "predicted":      self.type_classes[type_pred_id],
                "confidence_pct": round(float(type_probs[type_pred_id]) * 100, 1),
                "distribution":   {
                    self.type_classes[i]: round(float(type_probs[i]) * 100, 1)
                    for i in range(len(self.type_classes))
                },
            },
            "secondary_disease_risk": {
                "thyroid_risk_pct":    round(float(risk_probs[0]) * 100, 1),
                "autoimmune_risk_pct": round(float(risk_probs[1]) * 100, 1),
                "thyroid_flag":        bool(risk_probs[0] > 0.50),
                "autoimmune_flag":     bool(risk_probs[1] > 0.50),
            },
            "sideeffect_risk": {
                "level": self.sidefx_classes[side_pred_id],
                "distribution": {
                    self.sidefx_classes[i]: round(float(side_probs[i]) * 100, 1)
                    for i in range(len(self.sidefx_classes))
                },
                "high_risk_flag": bool(side_probs[2] > 0.40),
            },
            "severity": {
                "predicted_score":  round(sev_clamp, 2),
                "uncertainty_95ci": [ci_lo, ci_hi],
                "band":             sev_label,
                "description":      sev_desc,
            },
            "composite_risk_score":    round(float(composite), 3),
            "clinical_interpretation": _composite_interpretation(composite),
            "modality_gates":          {k: round(v, 3) for k, v in gates.items()},
        }
