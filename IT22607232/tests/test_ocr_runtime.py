from IT22607232.app.ocr_runtime import build_ocr_result, extract_labs_from_text


def test_extract_labs_from_text_reads_core_risk_fields():
    text = """
    C-REACTIVE PROTEIN 12.8 mg/L
    FREE THYROXINE (FT4) 1.14 ng/dL
    IMMUNOGLOBULIN E (IgE LEVEL) 455.60 IU/mL
    VITAMIN D 22.68 ng/mL
    AGE 34
    """

    result = extract_labs_from_text(text)

    assert result["CRP"] == 12.8
    assert result["FT4"] == 1.14
    assert result["IgE"] == 455.60
    assert result["VitD"] == 22.68
    assert result["Age"] == 34.0
    assert result["flags"]["missing"] == []


def test_extract_labs_from_text_marks_missing_values():
    text = "Only random clinical note text without measurable lab values."

    result = extract_labs_from_text(text)

    assert result["CRP"] is None
    assert result["FT4"] is None
    assert result["IgE"] is None
    assert result["VitD"] is None
    assert result["Age"] is None
    assert set(result["flags"]["missing"]) == {"CRP", "FT4", "IgE", "VitD", "Age"}


def test_build_ocr_result_keeps_audit_information():
    lab_texts = [
        "CRP 10.2 mg/L\nVitamin D 18.0 ng/mL",
        "IgE 220 IU/mL\nFT4 0.95 ng/dL\nAge 29",
    ]
    rx_texts = [
        "Cetirizine 10 mg noct\nPrednisolone 5 mg bd"
    ]

    result = build_ocr_result(lab_texts, rx_texts)

    assert result["labs_extracted"]["CRP"] == 10.2
    assert result["labs_extracted"]["VitD"] == 18.0
    assert result["labs_extracted"]["IgE"] == 220.0
    assert result["labs_extracted"]["FT4"] == 0.95
    assert result["labs_extracted"]["Age"] == 29.0
    assert "crp" in result["investigations_raw"].lower()
    assert "cetirizine" in result["symptoms_raw"].lower()
    assert result["missing_fields"] == []
