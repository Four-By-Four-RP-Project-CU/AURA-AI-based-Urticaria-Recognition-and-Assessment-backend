import unittest

from IT22577160.app.model_runtime import classify_uas7


class UAS7ClassificationTests(unittest.TestCase):
    def test_classify_uas7_returns_expected_mild_band(self):
        result = classify_uas7(10)

        self.assertEqual(result["category"], "Mild urticaria")
        self.assertEqual(result["recommended_step"], "STEP_1")

    def test_classify_uas7_returns_expected_severe_band(self):
        result = classify_uas7(35)

        self.assertEqual(result["category"], "Severe urticaria")
        self.assertEqual(result["recommended_step"], "STEP_3")

    def test_classify_uas7_handles_out_of_range_values(self):
        result = classify_uas7(100)

        self.assertEqual(result["category"], "Out of range")
        self.assertIsNone(result["recommended_step"])

