"""
main.py — FastAPI entrypoint (AWS + S3)

What this file does:
- Retrieve bucket location and artifact names from environment variables (set in the ECS Task Definition).
- Download pickled artifacts at startup (scaler, encoders, models) from S3 directly into memory.
- Store these objects on app.state so routers can use them.
- In CI/test mode (APP_ENV="test"), skip S3 entirely so smoke tests (/ and /ping) run offline.
"""

import os
import io
import joblib
import boto3
from fastapi import FastAPI
from routers import landing, ping as ping_router, predict_late, predict_very_late
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# Environment flag:
# - Default (no APP_ENV) → "prod"
# - Tests/CI (APP_ENV="test") → skip S3, use placeholders
APP_ENV = os.getenv("APP_ENV", "prod")
IS_TEST = (APP_ENV == "test")


# ─────────────────────────────────────────────
# Create the FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Shipment Delay Prediction API",
    description="FastAPI on ECS Fargate — artifacts loaded from S3 at startup.",
    version="1.0.0",
)

# ─────────────────────────────────────────────
# Retrieve bucket location and artifact names from environment variables
# (only enforced in non-test mode)
# ─────────────────────────────────────────────
BUCKET = os.environ.get("S3_BUCKET")
SCALER_KEY = os.environ.get("SCALER_KEY")
ONEHOT_KEY = os.environ.get("ONEHOT_KEY")
ORDINAL_KEY = os.environ.get("ORDINAL_KEY")
LATE_KEY = os.environ.get("LATE_MODEL_KEY")
VERY_LATE_KEY = os.environ.get("VERY_LATE_MODEL_KEY")

# Check that all required environment variables are set
if not IS_TEST:
    required_vars = {
        "ARTIFACT_BUCKET": BUCKET,
        "SCALER_KEY": SCALER_KEY,
        "ONEHOT_KEY": ONEHOT_KEY,
        "ORDINAL_KEY": ORDINAL_KEY,
        "LATE_MODEL_KEY": LATE_KEY,
        "VERY_LATE_MODEL_KEY": VERY_LATE_KEY,
    }

    missing = []
    for name, value in required_vars.items():
        if not value:
            missing.append(name)

    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))


# ─────────────────────────────────────────────
#  Create S3 client and helper function to load artifacts from bucket
# ─────────────────────────────────────────────
s3 = None if IS_TEST else boto3.client("s3")

def load_joblib_from_s3(bucket: str, key: str):
    """
    Download a .pkl file from S3 into memory and return the loaded Python object.
    Avoids writing to disk by streaming into an in-memory buffer.
    """
    if s3 is None:
        raise RuntimeError("S3 client is not initialized (test mode?)")
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return joblib.load(buf)


# ─────────────────────────────────────────────
# App startup: load all artifacts once and keep them in memory
# Routers will access them via request.app.state.<name>
# ─────────────────────────────────────────────
@app.on_event("startup")
def load_artifacts() -> None:
    if IS_TEST:
        log.info("APP_ENV=test: skipping S3 artifact loading")
        return
    
    try:
        app.state.scaler = load_joblib_from_s3(BUCKET, SCALER_KEY)
        app.state.onehot = load_joblib_from_s3(BUCKET, ONEHOT_KEY)
        app.state.ordinal = load_joblib_from_s3(BUCKET, ORDINAL_KEY)
        app.state.late_model = load_joblib_from_s3(BUCKET, LATE_KEY)
        app.state.very_late_model = load_joblib_from_s3(BUCKET, VERY_LATE_KEY)
        log.info("Loaded artifacts from S3 successfully")
        
    except Exception as e:
        log.error(f"Failed to load artifacts from S3: {e}")
        raise


# ─────────────────────────────────────────────
# Mount routers
# ─────────────────────────────────────────────
app.include_router(landing.router)
app.include_router(ping_router.router)
app.include_router(predict_late.router)
app.include_router(predict_very_late.router)