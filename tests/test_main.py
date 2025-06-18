"""
tests/test_main.py

Basic integration tests for core API endpoints:
- Landing page (`/`)
- Health check (`/ping`)

These tests verify that the root URL returns an HTML landing page,
and that the health check endpoint responds with status 200 and expected JSON.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Late Shipment Prediction API" in response.text

def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}