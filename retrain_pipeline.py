"""
retrain_pipeline.py

Orchestrated retraining pipeline for late shipment prediction models.

This script defines a Prefect flow that automates the full lifecycle:
1. Load raw shipment data
2. Clean and validate the data
3. Engineer predictive features
4. Preprocess features (split, encode, scale)
5. Train and evaluate the "late" and "very late" Random Forest models
6. Save trained artifacts locally and optionally upload to S3
7. Log results to MLflow for experiment tracking
8. Send notification on success or failure
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import sys
import os
import datetime
from pathlib import Path
from prefect import flow


# ─────────────────────────────────────────────
# CONFIG: Paths (resolve base_dir and make src importable)
# ─────────────────────────────────────────────
try:
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path().resolve()

# Ensure the src/ folder is on Python's import path so local modules can be found
src_dir = (base_dir / "src").resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
    
raw_data_file = base_dir / "data" / "raw" / "shipments_raw.csv"
unprocessed_data_dir = base_dir / "data" / "unprocessed"
preprocessed_data_dir = base_dir / "data" / "preprocessed"
late_model_file = base_dir / "models" / "late_model.pkl"
very_late_model_file = base_dir / "models" / "very_late_model.pkl"
scaler_file = base_dir / "models" / "scaler.pkl"
onehot_encoder_file = base_dir / "models" / "onehot_encoder.pkl"
ordinal_encoder_file = base_dir / "models" / "ordinal_encoder.pkl"


# ─────────────────────────────────────────────
# CONFIG: Environment (safe defaults)
# ─────────────────────────────────────────────
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_MODELS_BASE = os.getenv("S3_MODELS_BASE", "models")
S3_PREPROC_BASE = os.getenv("S3_PREPROC_BASE", "preprocessing")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
from src.logger import get_logger
logger = get_logger(__name__)


# ─────────────────────────────────────────────
# TASKS (Prefect task wrappers)
# ─────────────────────────────────────────────
from src.tasks import (
    t_load_raw_data,
    t_clean,
    t_engineer,
    t_preprocess,
    t_train_late,
    t_train_very_late,
    t_upload_models,
    t_notify,
)

