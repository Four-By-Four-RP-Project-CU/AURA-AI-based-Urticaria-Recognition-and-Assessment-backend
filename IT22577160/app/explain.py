from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

def overlay_heatmap_on_pil(image_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img = np.array(image_pil.convert("RGB"))
    hm = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    hm = np.uint8(255 * np.clip(hm, 0, 1))
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 1 - alpha, hm_color[:, :, ::-1], alpha, 0)  # BGR->RGB fix
    return Image.fromarray(out)

def redness_map(image_pil: Image.Image) -> np.ndarray:
    """
    Simple erythema proxy: high 'a' channel in LAB color space.
    Returns heatmap in [0,1].
    """
    img = np.array(image_pil.convert("RGB"))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1].astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return a

class GradCAM:
    """
    Minimal Grad-CAM for timm EfficientNet backbone:
    We hook the last conv-like block by name heuristic.
    """
    def __init__(self, model, target_module_name: str):
        self.model = model
        self.target_module_name = target_module_name
        self.activations = None
        self.gradients = None

        module = dict([*model.named_modules()]).get(target_module_name)
        if module is None:
            raise ValueError(f"Target module not found: {target_module_name}")

        module.register_forward_hook(self._forward_hook)
        module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def compute(self, image_tensor, lab_tensor, clin_tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        # Model returns a dict (drug_logits, step_logits, gate_w, …)
        out = self.model(image_tensor, lab_tensor, clin_tensor)
        drug_logits = out["drug_logits"] if isinstance(out, dict) else out[0]

        score = drug_logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        A = self.activations
        G = self.gradients
        # Guard: activations must be 4-D [B, C, H, W]
        if A is None or G is None or A.dim() != 4:
            return np.zeros((image_tensor.shape[-2], image_tensor.shape[-1]), dtype=np.float32)

        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=False)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam.detach().cpu().numpy()[0]


# ──────────────────────────────────────────────────────────────────────────────
# CU-specific morphological image analysis
# ──────────────────────────────────────────────────────────────────────────────

def compute_cu_characteristics(image_pil: "Image.Image") -> dict:
    """
    Derive Chronic Urticaria–specific image features from a skin photograph.

    Pipeline
    --------
    • Erythema / redness  — CIE LAB a*-channel analysis (positive a* = red/pink).
    • Wheal detection     — binary redness mask → morphological cleanup → contour extraction.
    • Per-wheal geometry  — diameter (% of image diagonal), circularity (4πA/P²),
                            aspect ratio (long/short ellipse axis).

    Returns a flat dict matching the keys expected by pdf_report._section_cu_characteristics().
    """
    img = np.array(image_pil.convert("RGB"))
    h, w = img.shape[:2]
    img_diag = float(np.sqrt(h * h + w * w))

    # ── ERYTHEMA (CIE LAB a* channel) ─────────────────────────────────────────
    # OpenCV stores a* in [0, 255]; 128 = neutral, >128 = reddish.
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    a_raw   = lab_img[:, :, 1].astype(np.float32)
    a_norm  = (a_raw - 128.0) / 127.0          # → [-1, +1]
    redness = np.clip(a_norm, 0.0, 1.0)        #  only positive (red) values

    redness_mean = float(np.mean(redness))
    redness_max  = float(np.max(redness))

    erythema_thr  = 0.12                        # pixels above this considered erythematous
    red_mask_bool = redness > erythema_thr
    coverage_pct  = float(100.0 * np.sum(red_mask_bool) / (redness.size + 1e-8))
    erythema_index = float(redness_mean * coverage_pct)  # composite ≈ 0–50

    # ── WHEAL DETECTION ────────────────────────────────────────────────────────
    redness_u8   = np.uint8(redness * 255)
    _, wheal_bin = cv2.threshold(redness_u8, int(erythema_thr * 255), 255, cv2.THRESH_BINARY)

    # Morphological cleanup: close gaps, then open to remove tiny spec noise
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
    wheal_bin = cv2.morphologyEx(wheal_bin, cv2.MORPH_CLOSE, k_close, iterations=2)
    wheal_bin = cv2.morphologyEx(wheal_bin, cv2.MORPH_OPEN,  k_open,  iterations=1)

    contours, _ = cv2.findContours(wheal_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Keep only contours large enough to be clinically relevant wheals
    min_area = (img_diag * 0.015) ** 2          # ≥ 1.5% of image diagonal
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    wheal_count  = len(valid_contours)
    diameters:    list = []
    circularities: list = []
    aspect_ratios: list = []

    for c in valid_contours:
        area      = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True) + 1e-8
        circ      = float(np.clip(4.0 * np.pi * area / (perimeter ** 2), 0.0, 1.0))
        circularities.append(circ)

        eq_diam = float(np.sqrt(4.0 * area / np.pi))
        diameters.append(100.0 * eq_diam / img_diag)

        if len(c) >= 5:
            try:
                _ell  = cv2.fitEllipse(c)
                axes  = sorted(_ell[1])
                ar    = float(axes[1] / (axes[0] + 1e-8))
            except Exception:
                xb, yb, wb, hb = cv2.boundingRect(c)
                ar = float(max(wb, hb) / (min(wb, hb) + 1e-8))
        else:
            xb, yb, wb, hb = cv2.boundingRect(c)
            ar = float(max(wb, hb) / (min(wb, hb) + 1e-8))
        aspect_ratios.append(max(ar, 1.0))

    avg_diam  = float(np.mean(diameters))    if diameters    else 0.0
    max_diam  = float(np.max(diameters))     if diameters    else 0.0
    mean_circ = float(np.mean(circularities)) if circularities else 0.0
    mean_ar   = float(np.mean(aspect_ratios)) if aspect_ratios else 1.0

    # ── Distribution pattern ───────────────────────────────────────────────────
    if wheal_count == 0:        distribution = "None detected"
    elif wheal_count == 1:      distribution = "Solitary lesion"
    elif wheal_count <= 3:      distribution = "Focal (2–3 lesions)"
    elif wheal_count <= 7:      distribution = "Scattered (4–7 lesions)"
    else:                       distribution = "Diffuse / confluent"

    # ── Shape description ───────────────────────────────────────────────────────
    if not valid_contours:
        shape_desc = "N/A"
    elif mean_circ >= 0.75:
        shape_desc = "Round-oval / classic urticarial"
    elif mean_circ >= 0.50:
        shape_desc = "Irregular-oval"
    else:
        shape_desc = "Irregular / linear"

    if valid_contours and mean_ar > 2.0:
        shape_desc += " / elongated"

    return {
        "redness_mean_score":      redness_mean,
        "redness_max_score":       redness_max,
        "redness_coverage_pct":    coverage_pct,
        "erythema_index":          erythema_index,
        "wheal_count":             wheal_count,
        "wheal_avg_diameter_pct":  avg_diam,
        "wheal_max_diameter_pct":  max_diam,
        "wheal_mean_circularity":  mean_circ,
        "wheal_mean_aspect_ratio": mean_ar,
        "distribution_pattern":    distribution,
        "shape_description":       shape_desc,
    }