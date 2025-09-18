# tests/integration/conftest.py
import importlib
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(scope="function")
def client(monkeypatch):
    # Make sure we're NOT in CI mode (so startup loads S3)
    monkeypatch.delenv("APP_ENV", raising=False)

    monkeypatch.setenv("ARTIFACT_BUCKET", "late-shipments-artifacts-bengt")
    monkeypatch.setenv("SCALER_KEY", "preprocessing/v2025-09-04/scaler.pkl")
    monkeypatch.setenv("ONEHOT_KEY", "preprocessing/v2025-09-04/onehot_encoder.pkl")
    monkeypatch.setenv("ORDINAL_KEY", "preprocessing/v2025-09-04/ordinal_encoder.pkl")
    monkeypatch.setenv("LATE_MODEL_KEY", "models/late_model/v2025-09-04/late_model.pkl")
    monkeypatch.setenv("VERY_LATE_MODEL_KEY", "models/very_late_model/v2025-09-04/very_late_model.pkl")

    # Import (or re-import) the app so startup runs with the env above
    mod = importlib.import_module("api.main")
    importlib.reload(mod)

    with TestClient(mod.app) as c:
        yield c
