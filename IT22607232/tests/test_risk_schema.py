from IT22607232.app.Risk_schema import PredictRequest, RiskProfileResponse


def test_predict_request_defaults_are_safe():
    request = PredictRequest()

    assert request.symptoms_raw == ""
    assert request.investigations_raw == ""
    assert request.categorical == {}
    assert request.CRP is None
    assert request.Age is None


def test_risk_profile_response_validates_expected_shape():
    response = RiskProfileResponse.model_validate(
        {
            "case_id": "AURA-1234",
            "mongo_persisted": True,
            "urticaria_type": {
                "predicted": "Spontaneous",
                "confidence_pct": 81.2,
                "distribution": {"Inducible": 8.0, "Mixed": 10.8, "Spontaneous": 81.2},
            },
            "secondary_disease_risk": {
                "thyroid_risk_pct": 65.0,
                "autoimmune_risk_pct": 44.0,
                "thyroid_flag": True,
                "autoimmune_flag": False,
            },
            "sideeffect_risk": {
                "level": "LOW",
                "distribution": {"LOW": 71.0, "MODERATE": 21.0, "HIGH": 8.0},
                "high_risk_flag": False,
            },
            "severity": {
                "predicted_score": 4.2,
                "uncertainty_95ci": [3.2, 5.1],
                "band": "MODERATE",
                "description": "Moderate disease burden",
            },
            "composite_risk_score": 0.41,
            "clinical_interpretation": "MODERATE - monitor progression.",
            "modality_gates": {"type": 0.5, "risk": 0.6, "side": 0.4, "sev": 0.55},
        }
    )

    assert response.case_id == "AURA-1234"
    assert response.severity.band == "MODERATE"
    assert response.sideeffect_risk.high_risk_flag is False
