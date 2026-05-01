import unittest

from fastapi import HTTPException

from IT22577160.app.main import (
    _build_clin_values,
    _build_risk_context_summary,
    _extract_handoff_labs,
    _parse_json_form,
)


class MainHelperTests(unittest.TestCase):
    def test_build_clin_values_defaults_missing_inputs_to_zero(self):
        values = _build_clin_values(None, 170.0, None, 25.0, 4.0)

        self.assertEqual(values["Weight"], 0.0)
        self.assertEqual(values["Height"], 170.0)
        self.assertEqual(values["Diagnosed at the age of"], 25.0)
        self.assertEqual(values["Itching score"], 4.0)

    def test_parse_json_form_returns_empty_dict_for_blank_payload(self):
        self.assertEqual(_parse_json_form("", "payload"), {})

    def test_parse_json_form_rejects_invalid_json(self):
        with self.assertRaises(HTTPException) as ctx:
            _parse_json_form("{bad json}", "payload")

        self.assertEqual(ctx.exception.status_code, 422)

    def test_extract_handoff_labs_prefers_explicit_payload(self):
        labs, source, received = _extract_handoff_labs(
            {"ocr_info": {"labs_extracted": {"CRP": 10}}},
            {"CRP": 22.5},
        )

        self.assertEqual(labs, {"CRP": 22.5})
        self.assertEqual(source, "extracted_labs_json")
        self.assertTrue(received)

    def test_extract_handoff_labs_reads_nested_risk_profile_labs(self):
        labs, source, received = _extract_handoff_labs(
            {"ocr_info": {"labs_extracted": {"IgE": 455.6}}},
            {},
        )

        self.assertEqual(labs, {"IgE": 455.6})
        self.assertEqual(source, "risk_profile.ocr_info.labs_extracted")
        self.assertTrue(received)

    def test_build_risk_context_summary_builds_cautions_and_note(self):
        summary, note = _build_risk_context_summary(
            {
                "urticaria_type": {"predicted": "Chronic Urticaria"},
                "severity": {"band": "MODERATE", "predicted_score": 5.2},
                "secondary_disease_risk": {
                    "thyroid_flag": True,
                    "autoimmune_flag": False,
                },
                "sideeffect_risk": {"level": "HIGH", "high_risk_flag": True},
                "composite_risk_score": 0.73,
                "clinical_interpretation": "Requires closer follow-up.",
            }
        )

        self.assertEqual(summary["severity_band"], "MODERATE")
        self.assertTrue(summary["high_sideeffect_flag"])
        self.assertIn("elevated thyroid-associated comorbidity risk", summary["cautions"])
        self.assertIn("high side-effect risk", summary["cautions"])
        self.assertIn("Requires closer follow-up.", note)

