import importlib
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _dummy_profile() -> dict:
    return {
        "urticaria_type": {
            "predicted": "Spontaneous",
            "confidence_pct": 82.5,
            "distribution": {
                "Inducible": 5.0,
                "Mixed": 12.5,
                "Spontaneous": 82.5,
            },
        },
        "secondary_disease_risk": {
            "thyroid_risk_pct": 72.0,
            "autoimmune_risk_pct": 61.0,
            "thyroid_flag": True,
            "autoimmune_flag": True,
        },
        "sideeffect_risk": {
            "level": "MODERATE",
            "distribution": {
                "LOW": 18.0,
                "MODERATE": 62.0,
                "HIGH": 20.0,
            },
            "high_risk_flag": False,
        },
        "severity": {
            "predicted_score": 6.4,
            "uncertainty_95ci": [4.8, 7.9],
            "band": "SEVERE",
            "description": "Significant burden; consider closer monitoring",
        },
        "composite_risk_score": 0.62,
        "clinical_interpretation": "HIGH - Close monitoring recommended.",
        "modality_gates": {
            "type": 0.55,
            "risk": 0.61,
            "side": 0.58,
            "sev": 0.52,
        },
    }


@pytest.fixture
def risk_main_module(monkeypatch):
    class DummyRuntime:
        def __init__(self, artifacts_dir: str):
            self.artifacts_dir = artifacts_dir
            self.device = "cpu"

        def predict(self, symptoms_raw, investigations_raw, labs, categorical):
            profile = _dummy_profile()
            profile["echo"] = {
                "symptoms_raw": symptoms_raw,
                "investigations_raw": investigations_raw,
                "labs": labs,
                "categorical": categorical,
            }
            return profile

    stub_runtime_module = types.ModuleType("IT22607232.app.Risk_model_runtime")
    stub_runtime_module.Runtime = DummyRuntime

    sys.modules.pop("IT22607232.app.Risk_main", None)
    monkeypatch.setitem(sys.modules, "IT22607232.app.Risk_model_runtime", stub_runtime_module)

    module = importlib.import_module("IT22607232.app.Risk_main")
    monkeypatch.setattr(module, "save_risk_result", lambda **kwargs: True)
    monkeypatch.setattr(module, "generate_case_id", lambda: "AURA-TESTCASE")
    return module


@pytest.fixture
def api_client(risk_main_module):
    return TestClient(risk_main_module.app)
