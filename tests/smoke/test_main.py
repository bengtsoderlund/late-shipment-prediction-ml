"""
tests/smoke/test_main.py

Smoke tests for core API endpoints (run in CI with APP_ENV=test):
- Landing page (`/`)
- Health check (`/ping`)

These tests verify that the root URL returns an HTML landing page,
and that the health check endpoint responds with status 200 and expected JSON.
"""

import os
import importlib
from fastapi.testclient import TestClient

def test_root():
    os.environ["APP_ENV"] = "test"  # ensure startup skips S3 during import
    mod = importlib.import_module("api.main")
    importlib.reload(mod)
    with TestClient(mod.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.text and len(response.text) > 0

def test_ping():
    os.environ["APP_ENV"] = "test"  # ensure startup skips S3 during import
    mod = importlib.import_module("api.main")
    importlib.reload(mod)
    with TestClient(mod.app) as client:
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}