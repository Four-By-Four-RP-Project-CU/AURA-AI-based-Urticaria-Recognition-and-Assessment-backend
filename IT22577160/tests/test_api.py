import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from IT22577160.app.main import app


def _make_test_png_bytes() -> bytes:
    image = Image.new("RGB", (16, 16), color=(255, 240, 240))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class PrescriptionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint_returns_ok(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("model", payload)

    @patch("IT22577160.app.main.extract_labs_from_images")
    def test_extract_labs_endpoint_returns_structured_result(self, mock_extract):
        mock_extract.return_value = {
            "CRP": 12.5,
            "FT4": 1.2,
            "IgE": 250.0,
            "VitD": 30.0,
            "Age": 28.0,
            "flags": {"missing": []},
        }

        files = [("lab_reports", ("report.png", _make_test_png_bytes(), "image/png"))]
        response = self.client.post("/extract/labs", files=files)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["extracted"]["CRP"], 12.5)
        self.assertEqual(payload["warnings"], [])

    @patch("IT22577160.app.main.save_prescription_result", return_value=False)
    @patch("IT22577160.app.main._build_analysis_artifacts")
    def test_analyze_endpoint_returns_prediction_payload(self, mock_build, _mock_save):
        mock_build.return_value = {
            "predicted_drug_group": "H1_ANTIHISTAMINE",
            "confidence": 0.91,
            "top3": [["H1_ANTIHISTAMINE", 0.91]],
            "mapped_guideline_step": "STEP_1",
            "guideline_step_detail": {},
            "abstain": False,
            "ood_flag": False,
            "ood_z": 0.12,
            "modality_gate_weights": [0.4, 0.3, 0.3],
            "used_features": {"CRP": 12.5},
            "lab_sources": {"CRP": "manual"},
            "extracted_labs": {"CRP": 12.5},
            "notes": [],
            "uas7_score": 7.0,
            "uas7_interpretation": {"category": "Mild urticaria"},
            "guideline_step_alignment": "aligned",
            "cu_characteristics": {"wheal_count": 2},
            "_skin_pil": None,
            "_gradcam_pil": Image.new("RGB", (8, 8)),
            "_redness_pil": Image.new("RGB", (8, 8)),
        }

        files = {
            "skin_image": ("skin.png", _make_test_png_bytes(), "image/png"),
        }
        data = {
            "CRP": "12.5",
            "Weight": "65",
            "Height": "170",
            "Itching_score": "4",
            "UAS7": "7",
        }
        response = self.client.post("/analyze", files=files, data=data)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predicted_drug_group"], "H1_ANTIHISTAMINE")
        self.assertEqual(payload["mapped_guideline_step"], "STEP_1")
        self.assertFalse(payload["mongo_persisted"])

    @patch("IT22577160.app.main.save_prescription_result", return_value=True)
    @patch("IT22577160.app.main._build_analysis_artifacts")
    def test_analyze_endpoint_triggers_persistence(self, mock_build, mock_save):
        mock_build.return_value = {
            "predicted_drug_group": "H1_ANTIHISTAMINE",
            "confidence": 0.91,
            "top3": [["H1_ANTIHISTAMINE", 0.91]],
            "mapped_guideline_step": "STEP_1",
            "guideline_step_detail": {},
            "abstain": False,
            "ood_flag": False,
            "ood_z": 0.12,
            "modality_gate_weights": [0.4, 0.3, 0.3],
            "used_features": {"CRP": 12.5},
            "lab_sources": {"CRP": "manual"},
            "extracted_labs": {"CRP": 12.5},
            "notes": [],
            "uas7_score": 7.0,
            "uas7_interpretation": {"category": "Mild urticaria"},
            "guideline_step_alignment": "aligned",
            "cu_characteristics": {"wheal_count": 2},
            "_skin_pil": None,
            "_gradcam_pil": Image.new("RGB", (8, 8)),
            "_redness_pil": Image.new("RGB", (8, 8)),
        }

        files = {
            "skin_image": ("skin.png", _make_test_png_bytes(), "image/png"),
        }
        data = {
            "case_id": "CASE-001",
            "patient_name": "Test Patient",
            "CRP": "12.5",
            "Weight": "65",
            "Height": "170",
            "Itching_score": "4",
            "UAS7": "7",
        }
        response = self.client.post("/analyze", files=files, data=data)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["mongo_persisted"])
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args.kwargs["case_id"], "CASE-001")
        self.assertEqual(mock_save.call_args.kwargs["patient_name"], "Test Patient")

    @patch("IT22577160.app.main.save_prescription_result", return_value=True)
    @patch("IT22577160.app.main._build_analysis_artifacts")
    def test_analyze_from_risk_endpoint_triggers_persistence(self, mock_build, mock_save):
        mock_build.return_value = {
            "predicted_drug_group": "ADVANCED_THERAPY",
            "confidence": 0.88,
            "top3": [["ADVANCED_THERAPY", 0.88]],
            "mapped_guideline_step": "STEP_3",
            "guideline_step_detail": {},
            "abstain": False,
            "ood_flag": False,
            "ood_z": 0.10,
            "modality_gate_weights": [0.5, 0.2, 0.3],
            "used_features": {"IgE": 455.6},
            "lab_sources": {"IgE": "ocr"},
            "extracted_labs": {"IgE": 455.6},
            "notes": [],
            "uas7_score": 28.0,
            "uas7_interpretation": {"category": "Severe urticaria"},
            "guideline_step_alignment": "aligned",
            "cu_characteristics": {"wheal_count": 4},
            "_skin_pil": None,
            "_gradcam_pil": Image.new("RGB", (8, 8)),
            "_redness_pil": Image.new("RGB", (8, 8)),
        }

        files = {
            "skin_image": ("skin.png", _make_test_png_bytes(), "image/png"),
        }
        data = {
            "case_id": "CASE-RISK-01",
            "patient_name": "Risk Patient",
            "risk_profile_json": (
                '{"case_id":"CASE-RISK-01","ocr_info":{"labs_extracted":{"IgE":455.6}},'
                '"severity":{"band":"SEVERE","predicted_score":8.4},'
                '"sideeffect_risk":{"level":"HIGH","high_risk_flag":true}}'
            ),
            "UAS7": "28",
        }
        response = self.client.post("/analyze/from-risk", files=files, data=data)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["mongo_persisted"])
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args.kwargs["case_id"], "CASE-RISK-01")
