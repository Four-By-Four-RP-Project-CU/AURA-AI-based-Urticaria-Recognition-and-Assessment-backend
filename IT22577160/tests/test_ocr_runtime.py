import unittest

from IT22577160.app.ocr_runtime import (
    extract_labs_from_text,
    extract_vitd,
    normalize_text,
    vitd_to_ng_ml,
)


class OCRRuntimeTests(unittest.TestCase):
    def test_normalize_text_standardizes_problematic_characters(self):
        text = "Vitamin μ D – level × 2"

        normalized = normalize_text(text)

        self.assertIn("u", normalized)
        self.assertIn("-", normalized)
        self.assertIn("x", normalized)

    def test_vitd_unit_conversion_handles_nmol(self):
        self.assertEqual(vitd_to_ng_ml(50.0, "nmol/L"), 20.0)

    def test_extract_vitd_reads_value_from_text(self):
        text = "Vitamin D 45.0 ng/mL"

        self.assertEqual(extract_vitd(text), 45.0)

    def test_extract_labs_from_text_extracts_core_values(self):
        text = """
        CRP: 12.5 mg/L
        FREE THYROXINE (F.T4) 1.10 pmol/L
        IMMUNOGLOBULIN E (IgE LEVEL) 455.60 IU/mL
        Vitamin D 30.0 ng/mL
        Age 32
        """

        extracted = extract_labs_from_text(text)

        self.assertEqual(extracted["CRP"], 12.5)
        self.assertEqual(extracted["FT4"], 1.10)
        self.assertEqual(extracted["IgE"], 455.60)
        self.assertEqual(extracted["VitD"], 30.0)
        self.assertEqual(extracted["Age"], 32.0)
        self.assertEqual(extracted["flags"]["missing"], [])

    def test_extract_labs_from_text_reports_missing_values(self):
        extracted = extract_labs_from_text("Only ESR 20 is available")

        self.assertIn("CRP", extracted["flags"]["missing"])
        self.assertIn("FT4", extracted["flags"]["missing"])
        self.assertIn("IgE", extracted["flags"]["missing"])
        self.assertIn("VitD", extracted["flags"]["missing"])
        self.assertIn("Age", extracted["flags"]["missing"])
