from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Documented accepted keys for the categorical questionnaire
# (must match exact dataset column names)
# ─────────────────────────────────────────────────────────────────────────────
_CATEGORICAL_EXAMPLE = {
    "Sex": "Female",
    "History of Chronic Urticaria": "Yes",
    "Symptoms Of Urticaria": "Wheals and Angioedema",
    "Duration of Symptoms of urticaria": "More than 6 months",
    "If Wheals are present": "Yes",
    "The shape of an individual wheal": "Round",
    "Size of a single Wheal": "Small (< 1 cm)",
    "No. of wheals": "More than 20",
    "Duration of wheal": "Less than 24 hours",
    "Location": "Generalized",
    "If  angioedema is present": "Yes",
    "Duration of angioedema": "Less than 24 hours",
    "Discomfort of Swelling": "Severe",
    "Affect of Swelling on Daily activities": "Significantly",
    "Angioedema affect on appearance": "Yes",
    "Overall affect of Swelling": "Moderate",
    "Which applies to your wheals/angioedema or both?": "Both",
    "Which of the following applies to your symptoms of urticaria?": "Spontaneous",
    "Which time of the day do the symptoms occur?": "Morning",
    "Symptoms of Autoinflamation:": "None",
    "Alpha Gal": "No",
    "Specify other allergy": "None",
    "Remission of Angioedema after discontinuation of the drug:": "Yes",
    "Family History of Urticaria": "No",
    "Family history of thyroid diseases": "Yes",
    "Family history of autoimmune diseases": "No",
}


# ─────────────────────────────────────────────────────────────────────────────
# Input schema  (POST /predict — pure JSON)
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    All clinical values supplied as JSON. Lab values not provided are imputed
    from the training-set median. Categorical keys not supplied default to 'Unknown'.
    """

    # ── Free-text inputs fed to Bio_ClinicalBERT ─────────────────────────────
    symptoms_raw:       str = Field("", description="Free-text symptom description")
    investigations_raw: str = Field("", description="Free-text lab / investigation notes")

    # ── Numeric lab values (OCR-extractable, all optional) ───────────────────
    CRP:    Optional[float] = Field(None, description="C-Reactive Protein (mg/L)")
    FT4:    Optional[float] = Field(None, description="Free Thyroxine (pmol/L or ng/dL)")
    IgE:    Optional[float] = Field(None, description="Total IgE (IU/mL)")
    VitD:   Optional[float] = Field(None, description="Vitamin D 25-OH (ng/mL)")

    # ── Demographic / anthropometric inputs ──────────────────────────────────
    Age:          Optional[float] = Field(None, description="Patient age (years)")
    Weight:       Optional[float] = Field(None, description="Weight (kg)")
    Height:       Optional[float] = Field(None, description="Height (cm)")
    diagnosed_age: Optional[float] = Field(None, description="Age at CU diagnosis")

    # ── Clinical questionnaire (26 categorical features) ─────────────────────
    # Keys must match exact dataset column names (see example below).
    categorical: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Questionnaire answers. Use exact dataset column names as keys.",
        json_schema_extra={"example": _CATEGORICAL_EXAMPLE},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas — mirrors notebook make_risk_profile() output
# ─────────────────────────────────────────────────────────────────────────────

class UrticariaTypeResult(BaseModel):
    predicted:        str
    confidence_pct:   float
    distribution:     Dict[str, float]   # {type_label: probability_%}


class SecondaryRiskResult(BaseModel):
    thyroid_risk_pct:    float
    autoimmune_risk_pct: float
    thyroid_flag:        bool
    autoimmune_flag:     bool


class SideEffectResult(BaseModel):
    level:          str                  # LOW / MODERATE / HIGH
    distribution:   Dict[str, float]     # {LOW: %, MODERATE: %, HIGH: %}
    high_risk_flag: bool


class SeverityResult(BaseModel):
    predicted_score:  float              # 0–10, clamped
    uncertainty_95ci: List[float]        # [lo, hi] both clamped to [0, 10]
    band:             str                # MILD / MODERATE / SEVERE / EXTREME
    description:      str


class RiskProfileResponse(BaseModel):
    urticaria_type:          UrticariaTypeResult
    secondary_disease_risk:  SecondaryRiskResult
    sideeffect_risk:         SideEffectResult
    severity:                SeverityResult
    composite_risk_score:    float           # 0–1
    clinical_interpretation: str
    modality_gates:          Dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# OCR endpoint extras
# ─────────────────────────────────────────────────────────────────────────────

class OcrInfo(BaseModel):
    """What was automatically extracted from the uploaded lab-report files."""
    files_processed:    int
    labs_extracted:     Dict[str, Optional[float]]   # CRP/FT4/IgE/VitD (None = not found)
    investigations_raw: str                          # cleaned lab-lines sent to BERT
    symptoms_raw:       str                          # cleaned prescription text sent to BERT
    missing_fields:     List[str]                    # keys not found in any image


class OcrRiskProfileResponse(RiskProfileResponse):
    """Full risk profile + OCR extraction audit trail."""
    ocr_info: OcrInfo