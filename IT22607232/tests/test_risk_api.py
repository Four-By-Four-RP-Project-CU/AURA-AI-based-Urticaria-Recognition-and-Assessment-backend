def test_predict_endpoint_returns_structured_profile(api_client):
    payload = {
        "symptoms_raw": "wheals with itching for several weeks",
        "investigations_raw": "CRP elevated, IgE elevated",
        "CRP": 12.3,
        "FT4": 1.1,
        "IgE": 320.0,
        "VitD": 24.5,
        "Age": 35,
        "categorical": {
            "Sex": "Female",
            "History of Chronic Urticaria": "Yes",
        },
    }

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["case_id"] == "AURA-TESTCASE"
    assert data["mongo_persisted"] is True
    assert data["urticaria_type"]["predicted"] == "Spontaneous"
    assert data["severity"]["band"] == "SEVERE"


def test_predict_ocr_rejects_invalid_categorical_json(api_client):
    response = api_client.post(
        "/predict-ocr",
        files=[],
        data={"categorical_json": "{not-valid-json"},
    )

    assert response.status_code == 422
    assert "categorical_json is not valid JSON" in response.json()["detail"]
