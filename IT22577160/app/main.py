import os, uuid, io, traceback, json
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .schemas import AnalyzeResponse, AnalyzeFromRiskResponse, ExtractLabsResponse
from .model_runtime import ModelRuntime, classify_uas7
from .ocr_runtime import extract_labs_from_images
from .explain import GradCAM, overlay_heatmap_on_pil, redness_map, compute_cu_characteristics
from .pdf_report import build_pdf_report

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "..", "artifacts"))
runtime = ModelRuntime(ARTIFACTS_DIR)

app = FastAPI(title="AURA CSU Decision Support API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    return JSONResponse(status_code=500, content={"detail": str(exc), "traceback": tb})


# ── Grad-CAM setup ─────────────────────────────────────────────────────────────
GRADCAM_TARGET = os.getenv("GRADCAM_TARGET", "image_backbone.conv_head")
gradcam = None
try:
    gradcam = GradCAM(runtime.model, GRADCAM_TARGET)
except Exception:
    gradcam = None


# ── Shared analysis helper ─────────────────────────────────────────────────────
def _build_analysis_artifacts(
    skin_pil: Image.Image,
    lab_reports_bytes: list,
    lab_overrides: Dict[str, Any],
    clin_values: Dict[str, float],
    uas7: Optional[float],
    daily_wheal_avg: Optional[float],
    daily_pruritus_avg: Optional[float],
    abstain_threshold: float,
    prefetched_labs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the full AURA analysis pipeline.

    Returns a dict that is a superset of AnalyzeResponse plus three private keys
    (_skin_pil, _gradcam_pil, _redness_pil) used only by /report/pdf.
    """
    # 1 — OCR lab extraction
    extracted_labs: Dict[str, Any] = dict(prefetched_labs or {})
    if not extracted_labs and lab_reports_bytes:
        extracted_labs = extract_labs_from_images(lab_reports_bytes)

    # 2 — Merge manual overrides with OCR (manual wins when non-None and non-zero).
    #     When a value is still unavailable after checking both sources, fall back
    #     to the training-set mean so the model receives a neutral z-score (≈ 0)
    #     rather than an extreme negative that distorts predictions.
    LAB_TRAIN_MEANS: Dict[str, float] = {
        "CRP":  10.19,
        "FT4":  1.306,
        "IgE":  395.5,
        "VitD": 32.37,
        "Age":  46.83,
    }

    def _pick(manual_val: Any, ocr_key: str) -> tuple:
        """Returns (value, source) where source is 'manual', 'ocr', or 'fallback'."""
        if manual_val is not None and float(manual_val) != 0.0:
            return float(manual_val), "manual"
        ocr_val = extracted_labs.get(ocr_key)
        if ocr_val is not None and float(ocr_val) != 0.0:
            return float(ocr_val), "ocr"
        return LAB_TRAIN_MEANS.get(ocr_key, 0.0), "fallback"  # training-mean fallback

    _crp,  _crp_src  = _pick(lab_overrides.get("CRP"),  "CRP")
    _ft4,  _ft4_src  = _pick(lab_overrides.get("FT4"),  "FT4")
    _ige,  _ige_src  = _pick(lab_overrides.get("IgE"),  "IgE")
    _vitd, _vitd_src = _pick(lab_overrides.get("VitD"), "VitD")
    _age,  _age_src  = _pick(lab_overrides.get("Age"),  "Age")

    lab_values: Dict[str, float] = {
        "CRP":  _crp,
        "FT4":  _ft4,
        "IgE":  _ige,
        "VitD": _vitd,
        "Age":  _age,
    }
    lab_sources: Dict[str, str] = {
        "CRP":  _crp_src,
        "FT4":  _ft4_src,
        "IgE":  _ige_src,
        "VitD": _vitd_src,
        "Age":  _age_src,
    }

    # 3 — Model prediction
    pred = runtime.predict(skin_pil, lab_values, clin_values, abstain_threshold=abstain_threshold)

    # 3b — OOD guard: reject non-CSU images before any further processing
    if pred.get("ood_flag"):
        ood_z = round(pred.get("ood_z", 0.0), 4)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "non_csu_image",
                "message": (
                    f"The uploaded image does not appear to be a Chronic Spontaneous "
                    f"Urticaria (CSU) skin photograph (prototype similarity distance "
                    f"{ood_z:.4f}, threshold {runtime.proto_thr or 0.7418:.4f}). "
                    f"The image features are outside the training distribution. "
                    f"Please upload a clear, close-up photo of the affected skin area."
                ),
                "ood_z": ood_z,
            },
        )

    # 4 — UAS7 scoring
    uas7_value: Optional[float] = None
    uas7_interp: Optional[Dict[str, Any]] = None
    step_alignment: Optional[str] = None

    if uas7 is not None:
        uas7_value = float(uas7)
    elif daily_wheal_avg is not None and daily_pruritus_avg is not None:
        uas7_value = round((float(daily_wheal_avg) + float(daily_pruritus_avg)) * 7, 1)
    elif clin_values.get("Itching score", 0.0) > 0:
        # Fallback: map Itching score (0–6 NRS) linearly to UAS7 (0–42)
        uas7_value = round(float(clin_values["Itching score"]) * 7, 1)

    if uas7_value is not None:
        uas7_interp = classify_uas7(uas7_value)
        pred_step   = pred["mapped_guideline_step"]
        uas_step    = uas7_interp.get("recommended_step")
        if uas_step is None:
            step_alignment = None
        elif pred_step == uas_step:
            step_alignment = "aligned"
        else:
            pred_num = int(pred_step.split("_")[1])
            uas_num  = int(uas_step.split("_")[1])
            step_alignment = "model_higher" if pred_num > uas_num else "model_lower"

    # 5 — Grad-CAM overlay
    notes: list = []
    gradcam_pil: Image.Image = skin_pil
    if gradcam is not None:
        try:
            img_t, lab_t, clin_t = runtime.preprocess(skin_pil, lab_values, clin_values)
            class_idx = runtime.drug_groups.index(pred["predicted_drug_group"])
            hm = gradcam.compute(img_t, lab_t, clin_t, class_idx)
            gradcam_pil = overlay_heatmap_on_pil(skin_pil, hm)
        except Exception as _gc_err:
            notes.append(f"Grad-CAM failed: {_gc_err!s}")
    else:
        notes.append("Grad-CAM not available — set GRADCAM_TARGET env var to a valid module name.")

    # 6 — Redness overlay
    red_hm      = redness_map(skin_pil)
    redness_pil = overlay_heatmap_on_pil(skin_pil, red_hm)

    # 7 — CU image characteristics (erythema profile + wheal geometry)
    cu_chars = compute_cu_characteristics(skin_pil)

    # 8 — Used-features dict
    used_features = {
        **{k: float(lab_values.get(k, 0.0)) for k in runtime.lab_cols},
        **{k: float(clin_values.get(k, 0.0)) for k in runtime.clin_cols},
    }

    return {
        **pred,
        "used_features":            used_features,
        "lab_sources":              lab_sources,
        "extracted_labs":           extracted_labs,
        "notes":                    notes,
        "uas7_score":               uas7_value,
        "uas7_interpretation":      uas7_interp,
        "guideline_step_alignment": step_alignment,
        "cu_characteristics":       cu_chars,
        # Private – stripped before AnalyzeResponse serialisation
        "_skin_pil":    skin_pil,
        "_gradcam_pil": gradcam_pil,
        "_redness_pil": redness_pil,
    }


def _build_clin_values(
    Weight: Optional[float],
    Height: Optional[float],
    Age_experienced_first_symptoms: Optional[float],
    Diagnosed_at_the_age_of: Optional[float],
    Itching_score: Optional[float],
) -> Dict[str, float]:
    return {
        "Weight":                                                          Weight or 0.0,
        "Height":                                                          Height or 0.0,
        "Age experienced the first symptoms":                              Age_experienced_first_symptoms or 0.0,
        "Diagnosed at the age of":                                         Diagnosed_at_the_age_of or 0.0,
        "Itching score":                                                   Itching_score or 0.0,
        "1.7 If angioedema is present,\nWhich of the following Drugs do you use": 0.0,
    }


def _parse_json_form(payload: Optional[str], field_name: str) -> Dict[str, Any]:
    if not payload or not payload.strip():
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail=f"{field_name} must be a JSON object.")
    return parsed


def _extract_handoff_labs(
    risk_profile_payload: Dict[str, Any],
    extracted_labs_payload: Dict[str, Any],
) -> tuple[Dict[str, Any], str, bool]:
    if extracted_labs_payload:
        return extracted_labs_payload, "extracted_labs_json", bool(risk_profile_payload)

    if isinstance(risk_profile_payload.get("ocr_info"), dict):
        labs = risk_profile_payload["ocr_info"].get("labs_extracted")
        if isinstance(labs, dict):
            return labs, "risk_profile.ocr_info.labs_extracted", True

    if isinstance(risk_profile_payload.get("extracted_labs"), dict):
        return risk_profile_payload["extracted_labs"], "risk_profile.extracted_labs", True

    return {}, "manual_only", bool(risk_profile_payload)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True, "device": str(runtime.device), "model": runtime.cfg["model_name"]}


@app.post("/extract/labs", response_model=ExtractLabsResponse)
async def extract_labs(lab_reports: list[UploadFile] = File(...)):
    blobs     = [await f.read() for f in lab_reports]
    extracted = extract_labs_from_images(blobs)
    warnings: list = []
    if extracted.get("flags", {}).get("missing"):
        warnings.append(f"Missing labs: {extracted['flags']['missing']}")
    return {"extracted": extracted, "warnings": warnings}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    skin_image:  UploadFile = File(...),
    lab_reports: list[UploadFile] = File(default=[]),

    CRP:  Optional[float] = Form(default=None),
    FT4:  Optional[float] = Form(default=None),
    IgE:  Optional[float] = Form(default=None),
    VitD: Optional[float] = Form(default=None),
    Age:  Optional[float] = Form(default=None),

    Weight:                         Optional[float] = Form(default=None),
    Height:                         Optional[float] = Form(default=None),
    Age_experienced_first_symptoms: Optional[float] = Form(default=None),
    Diagnosed_at_the_age_of:        Optional[float] = Form(default=None),
    Itching_score:                  Optional[float] = Form(default=None),

    UAS7: Optional[float] = Form(default=None,
        description="Urticaria Activity Score over 7 days (0-42). Provide directly or via per-day components."),
    daily_wheal_avg:    Optional[float] = Form(default=None,
        description="Average daily wheal score (0-3) over 7 days."),
    daily_pruritus_avg: Optional[float] = Form(default=None,
        description="Average daily pruritus score (0-3) over 7 days."),

    abstain_threshold: float = Form(default=0.55),
):
    img_bytes         = await skin_image.read()
    skin_pil          = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    lab_reports_bytes = [await f.read() for f in lab_reports] if lab_reports else []

    clin_values = _build_clin_values(
        Weight, Height, Age_experienced_first_symptoms, Diagnosed_at_the_age_of, Itching_score
    )

    result = _build_analysis_artifacts(
        skin_pil, lab_reports_bytes,
        {"CRP": CRP, "FT4": FT4, "IgE": IgE, "VitD": VitD, "Age": Age},
        clin_values,
        UAS7, daily_wheal_avg, daily_pruritus_avg,
        abstain_threshold,
    )
    result.pop("_skin_pil",    None)
    result.pop("_gradcam_pil", None)
    result.pop("_redness_pil", None)
    return result


@app.post("/analyze/from-risk", response_model=AnalyzeFromRiskResponse)
async def analyze_from_risk(
    skin_image: UploadFile = File(...),

    risk_profile_json: Optional[str] = Form(
        default=None,
        description="Full JSON response from the risk-analysis step. Used to reuse extracted lab values without uploading reports again.",
    ),
    extracted_labs_json: Optional[str] = Form(
        default=None,
        description="Optional JSON object of extracted lab values. Overrides labs found inside risk_profile_json when provided.",
    ),

    CRP: Optional[float] = Form(default=None),
    FT4: Optional[float] = Form(default=None),
    IgE: Optional[float] = Form(default=None),
    VitD: Optional[float] = Form(default=None),
    Age: Optional[float] = Form(default=None),

    Weight: Optional[float] = Form(default=None),
    Height: Optional[float] = Form(default=None),
    Age_experienced_first_symptoms: Optional[float] = Form(default=None),
    Diagnosed_at_the_age_of: Optional[float] = Form(default=None),
    Itching_score: Optional[float] = Form(default=None),

    UAS7: Optional[float] = Form(default=None),
    daily_wheal_avg: Optional[float] = Form(default=None),
    daily_pruritus_avg: Optional[float] = Form(default=None),

    abstain_threshold: float = Form(default=0.55),
):
    img_bytes = await skin_image.read()
    skin_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    risk_payload = _parse_json_form(risk_profile_json, "risk_profile_json")
    extracted_labs_payload = _parse_json_form(extracted_labs_json, "extracted_labs_json")
    handoff_labs, handoff_source, risk_profile_received = _extract_handoff_labs(
        risk_payload, extracted_labs_payload
    )

    clin_values = _build_clin_values(
        Weight, Height, Age_experienced_first_symptoms, Diagnosed_at_the_age_of, Itching_score
    )

    result = _build_analysis_artifacts(
        skin_pil=skin_pil,
        lab_reports_bytes=[],
        lab_overrides={"CRP": CRP, "FT4": FT4, "IgE": IgE, "VitD": VitD, "Age": Age},
        clin_values=clin_values,
        uas7=UAS7,
        daily_wheal_avg=daily_wheal_avg,
        daily_pruritus_avg=daily_pruritus_avg,
        abstain_threshold=abstain_threshold,
        prefetched_labs=handoff_labs,
    )
    result.pop("_skin_pil", None)
    result.pop("_gradcam_pil", None)
    result.pop("_redness_pil", None)
    result["handoff_source"] = handoff_source
    result["risk_profile_received"] = risk_profile_received
    result["reused_extracted_labs"] = bool(handoff_labs)
    return result


@app.post("/report/pdf")
async def report_pdf(
    skin_image:  UploadFile = File(...),
    lab_reports: list[UploadFile] = File(default=[]),

    case_id:      str = Form(default="",
        description="Case / session identifier. Auto-generated if blank."),
    patient_name: str = Form(default="",
        description="Patient name or anonymised label."),

    CRP:  Optional[float] = Form(default=None),
    FT4:  Optional[float] = Form(default=None),
    IgE:  Optional[float] = Form(default=None),
    VitD: Optional[float] = Form(default=None),
    Age:  Optional[float] = Form(default=None),

    Weight:                         Optional[float] = Form(default=None),
    Height:                         Optional[float] = Form(default=None),
    Age_experienced_first_symptoms: Optional[float] = Form(default=None),
    Diagnosed_at_the_age_of:        Optional[float] = Form(default=None),
    Itching_score:                  Optional[float] = Form(default=None),

    UAS7:               Optional[float] = Form(default=None),
    daily_wheal_avg:    Optional[float] = Form(default=None),
    daily_pruritus_avg: Optional[float] = Form(default=None),

    abstain_threshold: float = Form(default=0.55),

    # Optional: pre-computed analysis JSON sent by the frontend after /analyze.
    # When provided the model is NOT re-run, guaranteeing the PDF matches the
    # dashboard exactly.  Image processing (Grad-CAM / redness) is still run
    # on the uploaded skin image so the PDF visuals are always fresh.
    cached_result: Optional[str] = Form(default=None),
):
    img_bytes = await skin_image.read()
    skin_pil  = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if cached_result:
        # ── Fast path: reuse the already-computed prediction ─────────────────
        result = json.loads(cached_result)

        # Re-run image processing so the PDF visuals are generated from the
        # actual uploaded bytes (Grad-CAM, redness map, CU characteristics).
        gradcam_pil: Image.Image = skin_pil
        if gradcam is not None:
            try:
                used = result.get("used_features", {})
                lab_vals_cached = {k: float(used.get(k, 0.0))
                                   for k in runtime.lab_cols}
                clin_vals_cached = {k: float(used.get(k, 0.0))
                                    for k in runtime.clin_cols}
                img_t, lab_t, clin_t = runtime.preprocess(
                    skin_pil, lab_vals_cached, clin_vals_cached)
                class_idx = runtime.drug_groups.index(
                    result["predicted_drug_group"])
                hm = gradcam.compute(img_t, lab_t, clin_t, class_idx)
                gradcam_pil = overlay_heatmap_on_pil(skin_pil, hm)
            except Exception:
                pass

        red_hm      = redness_map(skin_pil)
        redness_pil = overlay_heatmap_on_pil(skin_pil, red_hm)

        # Refresh CU characteristics from this upload (keeps visual section
        # consistent; non-determinism here is purely cosmetic).
        result["cu_characteristics"] = compute_cu_characteristics(skin_pil)

    else:
        # ── Slow path: run full inference (fallback / direct API calls) ──────
        lab_reports_bytes = [await f.read() for f in lab_reports] if lab_reports else []

        clin_values = _build_clin_values(
            Weight, Height, Age_experienced_first_symptoms, Diagnosed_at_the_age_of, Itching_score
        )

        result = _build_analysis_artifacts(
            skin_pil, lab_reports_bytes,
            {"CRP": CRP, "FT4": FT4, "IgE": IgE, "VitD": VitD, "Age": Age},
            clin_values,
            UAS7, daily_wheal_avg, daily_pruritus_avg,
            abstain_threshold,
        )

        gradcam_pil = result.pop("_gradcam_pil", skin_pil)
        redness_pil = result.pop("_redness_pil", skin_pil)
        result.pop("_skin_pil", None)

    cid = case_id.strip() or str(uuid.uuid4())[:8].upper()
    patient_meta = {
        "Case ID": cid,
        "Patient": patient_name.strip() or "Anonymous",
    }

    pdf_bytes = build_pdf_report(
        patient_meta       = patient_meta,
        prediction         = result,
        extracted_labs     = result.get("extracted_labs", {}),
        images             = {"skin": skin_pil, "gradcam": gradcam_pil, "redness": redness_pil},
        cu_characteristics = result.get("cu_characteristics"),
    )

    return Response(
        content    = pdf_bytes,
        media_type = "application/pdf",
        headers    = {"Content-Disposition": f"attachment; filename=AURA_CSU_Report_{cid}.pdf"},
    )


@app.post("/report/pdf/from-risk")
async def report_pdf_from_risk(
    skin_image: UploadFile = File(...),

    case_id: str = Form(default=""),
    patient_name: str = Form(default=""),

    risk_profile_json: Optional[str] = Form(
        default=None,
        description="Full JSON response from the risk-analysis step. Used to reuse extracted lab values without re-uploading reports.",
    ),
    extracted_labs_json: Optional[str] = Form(
        default=None,
        description="Optional JSON object of extracted lab values. Overrides labs found inside risk_profile_json when provided.",
    ),
    cached_result: Optional[str] = Form(default=None),

    CRP: Optional[float] = Form(default=None),
    FT4: Optional[float] = Form(default=None),
    IgE: Optional[float] = Form(default=None),
    VitD: Optional[float] = Form(default=None),
    Age: Optional[float] = Form(default=None),

    Weight: Optional[float] = Form(default=None),
    Height: Optional[float] = Form(default=None),
    Age_experienced_first_symptoms: Optional[float] = Form(default=None),
    Diagnosed_at_the_age_of: Optional[float] = Form(default=None),
    Itching_score: Optional[float] = Form(default=None),

    UAS7: Optional[float] = Form(default=None),
    daily_wheal_avg: Optional[float] = Form(default=None),
    daily_pruritus_avg: Optional[float] = Form(default=None),

    abstain_threshold: float = Form(default=0.55),
):
    img_bytes = await skin_image.read()
    skin_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if cached_result:
        result = json.loads(cached_result)

        gradcam_pil: Image.Image = skin_pil
        if gradcam is not None:
            try:
                used = result.get("used_features", {})
                lab_vals_cached = {k: float(used.get(k, 0.0)) for k in runtime.lab_cols}
                clin_vals_cached = {k: float(used.get(k, 0.0)) for k in runtime.clin_cols}
                img_t, lab_t, clin_t = runtime.preprocess(skin_pil, lab_vals_cached, clin_vals_cached)
                class_idx = runtime.drug_groups.index(result["predicted_drug_group"])
                hm = gradcam.compute(img_t, lab_t, clin_t, class_idx)
                gradcam_pil = overlay_heatmap_on_pil(skin_pil, hm)
            except Exception:
                pass

        red_hm = redness_map(skin_pil)
        redness_pil = overlay_heatmap_on_pil(skin_pil, red_hm)
        result["cu_characteristics"] = compute_cu_characteristics(skin_pil)
    else:
        risk_payload = _parse_json_form(risk_profile_json, "risk_profile_json")
        extracted_labs_payload = _parse_json_form(extracted_labs_json, "extracted_labs_json")
        handoff_labs, _, _ = _extract_handoff_labs(risk_payload, extracted_labs_payload)
        clin_values = _build_clin_values(
            Weight, Height, Age_experienced_first_symptoms, Diagnosed_at_the_age_of, Itching_score
        )

        result = _build_analysis_artifacts(
            skin_pil=skin_pil,
            lab_reports_bytes=[],
            lab_overrides={"CRP": CRP, "FT4": FT4, "IgE": IgE, "VitD": VitD, "Age": Age},
            clin_values=clin_values,
            uas7=UAS7,
            daily_wheal_avg=daily_wheal_avg,
            daily_pruritus_avg=daily_pruritus_avg,
            abstain_threshold=abstain_threshold,
            prefetched_labs=handoff_labs,
        )

        gradcam_pil = result.pop("_gradcam_pil", skin_pil)
        redness_pil = result.pop("_redness_pil", skin_pil)
        result.pop("_skin_pil", None)

    cid = case_id.strip() or str(uuid.uuid4())[:8].upper()
    patient_meta = {
        "Case ID": cid,
        "Patient": patient_name.strip() or "Anonymous",
    }

    pdf_bytes = build_pdf_report(
        patient_meta=patient_meta,
        prediction=result,
        extracted_labs=result.get("extracted_labs", {}),
        images={"skin": skin_pil, "gradcam": gradcam_pil, "redness": redness_pil},
        cu_characteristics=result.get("cu_characteristics"),
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=AURA_CSU_Report_{cid}.pdf"},
    )
