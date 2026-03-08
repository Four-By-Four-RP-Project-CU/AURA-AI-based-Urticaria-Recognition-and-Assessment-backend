import json
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .Risk_schema import (
    PredictRequest,
    RiskProfileResponse,
    OcrRiskProfileResponse,
)
from .Risk_model_runtime import Runtime
from .ocr_runtime import process_upload, build_ocr_result

app = FastAPI(
    title="AURA – Chronic Urticaria Risk Profiling API",
    version="2.0",
    openapi_version="3.0.3",   # Force 3.0 so Swagger UI renders UploadFile as a file-picker (3.1 uses contentMediaType which Swagger UI shows as text)
    description=(
        "Multi-task learning model for CU urticaria type classification, "
        "secondary disease risk, side-effect risk, and symptom severity.\n\n"
        "### Endpoints\n"
        "| Endpoint | When to use |\n"
        "|---|---|\n"
        "| `POST /predict` | All values typed manually as JSON |\n"
        "| `POST /predict-ocr` | Upload lab-report images (OCR extracts CRP/FT4/IgE/VitD) + type demographics & questionnaire |\n\n"
        "### Categorical questionnaire keys\n"
        "The `categorical` / `categorical_json` field accepts any subset of these exact column names:\n\n"
        "**Family history:** `Family History of Urticaria`, `Family history of thyroid diseases`, "
        "`Family history of autoimmune diseases`\n\n"
        "**Questionnaire:** `Sex`, `History of Chronic Urticaria`, `Symptoms Of Urticaria`, "
        "`Duration of Symptoms of urticaria`, `If Wheals are present`, `The shape of an individual wheal`, "
        "`Size of a single Wheal`, `No. of wheals`, `Duration of wheal`, `Location`, "
        "`If  angioedema is present`, `Duration of angioedema`, `Discomfort of Swelling`, "
        "`Affect of Swelling on Daily activities`, `Angioedema affect on appearance`, "
        "`Overall affect of Swelling`, `Which applies to your wheals/angioedema or both?`, "
        "`Which of the following applies to your symptoms of urticaria?`, "
        "`Which time of the day do the symptoms occur?`, `Symptoms of Autoinflamation:`, "
        "`Alpha Gal`, `Specify other allergy`, "
        "`Remission of Angioedema after discontinuation of the drug:`"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rt = Runtime(artifacts_dir=str(Path(__file__).resolve().parents[1] / "artifacts"))


# ── Fix Swagger UI file-picker rendering ──────────────────────────────────────
# FastAPI 0.99+ emits OpenAPI 3.1 which uses {"contentMediaType":"application/octet-stream"}
# for UploadFile fields.  Swagger UI 4/5 does NOT render that as a file-picker button;
# it only does so for {"format":"binary"}.  Walk the generated schema and patch it.
def _fix_upload_schema(node: object) -> None:
    if isinstance(node, dict):
        if node.get("type") == "string" and node.get("contentMediaType") == "application/octet-stream":
            node.pop("contentMediaType")
            node["format"] = "binary"
        for v in node.values():
            _fix_upload_schema(v)
    elif isinstance(node, list):
        for item in node:
            _fix_upload_schema(item)


_original_openapi = app.openapi


def _patched_openapi() -> dict:
    schema = _original_openapi()
    _fix_upload_schema(schema)
    return schema


app.openapi = _patched_openapi  # type: ignore[method-assign]

def health():
    return {"status": "ok", "model": "GatedFusionMTL v2", "device": rt.device}


# ═════════════════════════════════════════════════════════════════════════════
#  Endpoint 1 — POST /predict
#  All values typed as JSON. OCR not involved.
# ═════════════════════════════════════════════════════════════════════════════

@app.post(
    "/predict",
    response_model=RiskProfileResponse,
    summary="Risk profile from typed values (no OCR)",
    tags=["Prediction"],
)
def predict(req: PredictRequest):
    """
    Submit all clinical values as JSON.

    - **Labs** (`CRP`, `FT4`, `IgE`, `VitD`, `Age`, `Weight`, `Height`, `diagnosed_age`):
      all optional — missing values are imputed from the training-set median.
    - **categorical**: dict of questionnaire answers using exact dataset column names.
      Missing keys default to `"Unknown"`. See API description for the full key list.
    - **symptoms_raw / investigations_raw**: free-text notes fed to Bio_ClinicalBERT.
    """
    labs = {
        "CRP":                      req.CRP,
        "FT4":                      req.FT4,
        "IgE":                      req.IgE,
        "VitD":                     req.VitD,
        "Age":                      req.Age,
        "Weight":                   req.Weight,
        "Height":                   req.Height,
        "Diagnosed at the age of":  req.diagnosed_age,
    }
    return rt.predict(
        symptoms_raw=req.symptoms_raw,
        investigations_raw=req.investigations_raw,
        labs=labs,
        categorical=req.categorical,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Endpoint 2 — POST /predict-ocr
#  Upload one or more lab-report images/PDFs.
#  OCR automatically reads CRP, FT4, IgE, VitD.
#  Type remaining clinical values via form fields.
# ═════════════════════════════════════════════════════════════════════════════

_ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"}


@app.post(
    "/predict-ocr",
    response_model=OcrRiskProfileResponse,
    summary="Risk profile from lab-report images + typed clinical values",
    tags=["Prediction"],
    responses={
        415: {"description": "Unsupported file type (use JPG / PNG / PDF)"},
        422: {"description": "categorical_json is not valid JSON"},
    },
)
async def predict_ocr(
    # ── Lab report images / PDFs ─────────────────────────────────────────────
    files: List[UploadFile] = File(
        default=[],
        description=(
            "One or more lab-report images or PDFs. "
            "OCR will extract CRP, FT4, IgE, VitD automatically. "
            "Supported: JPG, PNG, BMP, TIFF, PDF."
        ),
    ),

    # ── Demographics — not on lab reports, always typed ──────────────────────
    age:           Annotated[Optional[float], Form(description="Patient age (years)")] = None,
    weight:        Annotated[Optional[float], Form(description="Weight (kg)")] = None,
    height:        Annotated[Optional[float], Form(description="Height (cm)")] = None,
    diagnosed_age: Annotated[Optional[float], Form(description="Age at CU diagnosis")] = None,

    # ── Manual lab overrides — typed if OCR cannot read a value ──────────────
    crp_manual:  Annotated[Optional[float], Form(description="CRP (mg/L) — overrides OCR value if provided")] = None,
    ft4_manual:  Annotated[Optional[float], Form(description="FT4 (pmol/L) — overrides OCR value if provided")] = None,
    ige_manual:  Annotated[Optional[float], Form(description="IgE (IU/mL) — overrides OCR value if provided")] = None,
    vitd_manual: Annotated[Optional[float], Form(description="VitD (ng/mL) — overrides OCR value if provided")] = None,

    # ── Optional extra free-text notes ───────────────────────────────────────
    symptoms_text:      Annotated[Optional[str], Form(description="Typed symptom notes (appended to OCR prescription text)")] = None,
    investigations_text: Annotated[Optional[str], Form(description="Typed investigation notes (appended to OCR lab-summary text)")] = None,

    # ── Questionnaire answers as a JSON string ────────────────────────────────
    # All 26 categorical features in one field to avoid cluttering form-data.
    # Provide a JSON object whose keys are exact dataset column names.
    # Any missing key defaults to "Unknown".
    categorical_json: Annotated[str, Form(
        description=(
            'JSON object of clinical questionnaire answers. All keys optional — missing → "Unknown".\n\n'
            'Minimal example:\n'
            '{"Sex":"Female","History of Chronic Urticaria":"Yes",'
            '"Family History of Urticaria":"No",'
            '"Family history of thyroid diseases":"Yes",'
            '"Family history of autoimmune diseases":"No",'
            '"If  angioedema is present":"Yes",'
            '"Alpha Gal":"No"}'
        )
    )] = "{}",
):
    # ── 1. Parse categorical JSON ─────────────────────────────────────────────
    try:
        categorical: dict = json.loads(categorical_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"categorical_json is not valid JSON: {exc}",
        )

    # ── 2. Run OCR on every uploaded file ─────────────────────────────────────
    all_lab_texts: List[str] = []
    all_rx_texts:  List[str] = []

    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in _ALLOWED_EXT:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"File '{upload.filename}' has unsupported type '{ext}'. "
                    f"Allowed: {sorted(_ALLOWED_EXT)}"
                ),
            )
        content = await upload.read()
        result = process_upload(upload.filename or f"file{ext}", content)
        all_lab_texts.extend(result["lab_texts"])
        all_rx_texts.extend(result["rx_texts"])

    # ── 3. Consolidate OCR output ─────────────────────────────────────────────
    ocr = build_ocr_result(all_lab_texts, all_rx_texts)

    # ── 4. Build lab dict: OCR values, then manual overrides win ─────────────
    labs: dict = {k: v for k, v in ocr["labs_extracted"].items() if v is not None}

    if age           is not None: labs["Age"]                     = age
    if weight        is not None: labs["Weight"]                  = weight
    if height        is not None: labs["Height"]                  = height
    if diagnosed_age is not None: labs["Diagnosed at the age of"] = diagnosed_age
    if crp_manual    is not None: labs["CRP"]                     = crp_manual
    if ft4_manual    is not None: labs["FT4"]                     = ft4_manual
    if ige_manual    is not None: labs["IgE"]                     = ige_manual
    if vitd_manual   is not None: labs["VitD"]                    = vitd_manual

    # ── 5. Merge text inputs ──────────────────────────────────────────────────
    symptoms_raw       = " ".join(filter(None, [ocr["symptoms_raw"],       symptoms_text])).strip()
    investigations_raw = " ".join(filter(None, [ocr["investigations_raw"], investigations_text])).strip()

    # ── 6. Predict ────────────────────────────────────────────────────────────
    profile = rt.predict(
        symptoms_raw=symptoms_raw,
        investigations_raw=investigations_raw,
        labs=labs,
        categorical=categorical,
    )

    # ── 7. Return risk profile + OCR audit trail ──────────────────────────────
    return {
        **profile,
        "ocr_info": {
            "files_processed":    len(files),
            "labs_extracted":     ocr["labs_extracted"],
            "investigations_raw": ocr["investigations_raw"],
            "symptoms_raw":       ocr["symptoms_raw"],
            "missing_fields":     ocr["missing_fields"],
        },
    }

