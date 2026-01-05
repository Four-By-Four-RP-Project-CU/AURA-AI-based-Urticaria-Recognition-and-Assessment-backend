# model_def.py
import torch
import torch.nn as nn
import timm


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim * 2),
        )

    def forward(self, feats, cond, return_params: bool = False):
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        out = feats * (1.0 + gamma) + beta
        if return_params:
            return out, gamma, beta
        return out


class GC_MuPeN(nn.Module):
    """
    Returns:
      forward(...) -> drug_logits, step_logits, gate_w3, film_stats
      forward_with_maps(...) -> drug_logits, step_logits, gate_w3, film_stats, feat_map
    """
    def __init__(
        self,
        lab_in_dim: int,
        clinical_in_dim: int,
        num_drugs: int,
        num_steps: int,
        dropout: float = 0.3,
        fusion_hidden: int = 512,
        fusion_out: int = 256,
    ):
        super().__init__()

        # IMPORTANT: No downloads in backend. We load trained weights from model.pt
        self.image_backbone = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
        self.img_feat_dim = self.image_backbone.num_features

        self.lab_in_dim = lab_in_dim
        self.clin_in_dim = clinical_in_dim

        self.lab_mlp = nn.Sequential(
            nn.Linear(lab_in_dim, 128) if lab_in_dim > 0 else nn.Identity(),
            nn.ReLU() if lab_in_dim > 0 else nn.Identity(),
            nn.Dropout(dropout) if lab_in_dim > 0 else nn.Identity(),
            nn.Linear(128, 128) if lab_in_dim > 0 else nn.Identity(),
        )

        self.clin_mlp = nn.Sequential(
            nn.Linear(clinical_in_dim, 128) if clinical_in_dim > 0 else nn.Identity(),
            nn.ReLU() if clinical_in_dim > 0 else nn.Identity(),
            nn.Dropout(dropout) if clinical_in_dim > 0 else nn.Identity(),
            nn.Linear(128, 128) if clinical_in_dim > 0 else nn.Identity(),
        )

        cond_dim = 128 if lab_in_dim > 0 else 0
        self.film = FiLM(cond_dim, self.img_feat_dim) if cond_dim > 0 else None

        fused_in_dim = self.img_feat_dim + (128 if lab_in_dim > 0 else 0) + (128 if clinical_in_dim > 0 else 0)

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

    def _image_features_and_map(self, image: torch.Tensor):
        # timm: forward_features gives [B,C,H,W] for CNNs
        feat_map = self.image_backbone.forward_features(image)  # [B,C,H,W]
        # pool to [B,F]
        pooled = feat_map.mean(dim=(2, 3))
        return pooled, feat_map

    def _compute_film_stats(self, img_before, img_after, gamma, beta):
        return {
            "enabled": True,
            "gamma_l2_mean": float(torch.norm(gamma, p=2, dim=-1).mean().item()),
            "beta_l2_mean":  float(torch.norm(beta,  p=2, dim=-1).mean().item()),
            "gamma_abs_mean": float(gamma.abs().mean().item()),
            "beta_abs_mean":  float(beta.abs().mean().item()),
            "image_delta_l2_mean": float(torch.norm((img_after - img_before), p=2, dim=-1).mean().item()),
        }

    def forward(self, image, lab_feats, clinical_feats):
        img_feats, _ = self._image_features_and_map(image)  # [B,F]

        lab_emb = None if self.lab_in_dim == 0 else self.lab_mlp(lab_feats)
        clin_emb = None if self.clin_in_dim == 0 else self.clin_mlp(clinical_feats)

        film_stats = {
            "enabled": False,
            "gamma_l2_mean": 0.0,
            "beta_l2_mean": 0.0,
            "gamma_abs_mean": 0.0,
            "beta_abs_mean": 0.0,
            "image_delta_l2_mean": 0.0,
        }

        if self.film is not None and lab_emb is not None:
            img_before = img_feats
            img_feats, gamma, beta = self.film(img_feats, lab_emb, return_params=True)
            film_stats = self._compute_film_stats(img_before, img_feats, gamma, beta)

        parts = [img_feats]
        if lab_emb is not None:
            parts.append(lab_emb)
        if clin_emb is not None:
            parts.append(clin_emb)

        fused_in_for_gate = torch.cat(parts, dim=-1)
        gate_w = self.gate(fused_in_for_gate)

        gated_components = [img_feats * gate_w[:, 0:1]]
        if lab_emb is not None:
            gated_components.append(lab_emb * gate_w[:, 1:2])
        if clin_emb is not None:
            gated_components.append(clin_emb * gate_w[:, 2:3])

        fused = self.fusion(torch.cat(gated_components, dim=-1))
        return self.drug_head(fused), self.step_head(fused), gate_w, film_stats

    def forward_with_maps(self, image, lab_feats, clinical_feats):
        img_feats, feat_map = self._image_features_and_map(image)

        # ✅ CRITICAL: keep gradients for Grad-CAM
        feat_map.retain_grad()

        lab_emb = None if self.lab_in_dim == 0 else self.lab_mlp(lab_feats)
        clin_emb = None if self.clin_in_dim == 0 else self.clin_mlp(clinical_feats)

        film_stats = {
            "enabled": False,
            "gamma_l2_mean": 0.0,
            "beta_l2_mean": 0.0,
            "gamma_abs_mean": 0.0,
            "beta_abs_mean": 0.0,
            "image_delta_l2_mean": 0.0,
        }

        if self.film is not None and lab_emb is not None:
            img_before = img_feats
            img_feats, gamma, beta = self.film(img_feats, lab_emb, return_params=True)
            film_stats = self._compute_film_stats(img_before, img_feats, gamma, beta)

        parts = [img_feats]
        if lab_emb is not None:
            parts.append(lab_emb)
        if clin_emb is not None:
            parts.append(clin_emb)

        fused_in_for_gate = torch.cat(parts, dim=-1)
        gate_w = self.gate(fused_in_for_gate)

        gated_components = [img_feats * gate_w[:, 0:1]]
        if lab_emb is not None:
            gated_components.append(lab_emb * gate_w[:, 1:2])
        if clin_emb is not None:
            gated_components.append(clin_emb * gate_w[:, 2:3])

        fused = self.fusion(torch.cat(gated_components, dim=-1))
        drug_logits = self.drug_head(fused)
        step_logits = self.step_head(fused)

        return drug_logits, step_logits, gate_w, film_stats, feat_map
