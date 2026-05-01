from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class AnalyzeResponse(BaseModel):
    case_id: Optional[str] = Field(
        default=None,
        description="Shared case identifier used to link risk and prescription records across MongoDB collections.",
    )
    mongo_persisted: bool = Field(
        default=False,
        description="Whether this prediction was successfully persisted to MongoDB.",
    )
    predicted_drug_group: str
    confidence: float
    top3: List[List[Any]]
    mapped_guideline_step: str
    guideline_step_detail: Dict[str, Any] = Field(
        default_factory=dict,
        description="EAACI standard CSU guideline content for the predicted step (label, indication, drugs, duration, notes).",
    )
    abstain: bool
    ood_flag: bool
    ood_z: float
    modality_gate_weights: List[float]
    used_features: Dict[str, float]
    lab_sources: Dict[str, str] = Field(
        default_factory=dict,
        description="Origin of each lab value: 'manual' (user input), 'ocr' (extracted from report image), or 'fallback' (training-set mean, OCR+manual both unavailable).",
    )
    extracted_labs: Dict[str, Any] = {}
    notes: List[str] = []
    # UAS7 fields — populated when UAS7 score is provided
    uas7_score: Optional[float] = Field(
        default=None,
        description="Urticaria Activity Score (7-day, 0–42) provided by the clinician.",
    )
    uas7_interpretation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="UAS7 severity category, label, recommended guideline step, and scale info.",
    )
    guideline_step_alignment: Optional[str] = Field(
        default=None,
        description=(
            "Agreement indicator between the model-predicted step and the UAS7-recommended step. "
            "Values: 'aligned', 'model_higher', 'model_lower', or null when UAS7 not provided."
        ),
    )
    cu_characteristics: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "CU-specific image morphological analysis derived from the skin photograph: "
            "erythema scores, redness coverage, wheal count, diameter, circularity, "
            "aspect ratio, distribution pattern, and shape description."
        ),
    )

class AnalyzeFromRiskResponse(AnalyzeResponse):
    handoff_source: str = Field(
        default="risk_profile",
        description="Source used to prefill prescription labs without re-uploading reports.",
    )
    risk_profile_received: bool = Field(
        default=False,
        description="Whether a full risk-analysis payload was supplied in the request.",
    )
    reused_extracted_labs: bool = Field(
        default=False,
        description="Whether extracted lab values from the previous risk-analysis step were reused.",
    )
    risk_context_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rule-based contextual summary derived from the risk-analysis JSON. Used for decision support only; does not change the prescription model weights.",
    )
    integrated_clinical_note: Optional[str] = Field(
        default=None,
        description="A combined interpretation that places the prescription recommendation alongside risk-analysis findings.",
    )

class ExtractLabsResponse(BaseModel):
    extracted: Dict[str, Any]
    warnings: List[str] = []

class ReportResponse(BaseModel):
    report_id: str
