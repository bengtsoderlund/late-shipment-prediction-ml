"""
retrain_pipeline.py

Orchestrated retraining pipeline for late shipment prediction models.

This script defines a Prefect flow that automates the full lifecycle:
1. Load raw shipment data
2. Clean and validate the data
3. Engineer predictive features
4. Preprocess features (train-test split, save encoder and scaler)
5-6. Train and saves the "late" and "very late" Random Forest models
7. Uploads preprocess artifacts and models to S3
8. Log results to MLflow for experiment tracking
9. Send notification on success or failure
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import sys
import os
import datetime
from pathlib import Path
from dotenv import load_dotenv
from prefect import flow, get_run_logger


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

# Local paths
raw_data_file = base_dir / "data" / "raw" / "shipments_raw.csv"
unprocessed_data_dir = base_dir / "data" / "unprocessed"
preprocessed_data_dir = base_dir / "data" / "preprocessed"
late_model_file = base_dir / "models" / "late_model.pkl"
very_late_model_file = base_dir / "models" / "very_late_model.pkl"
scaler_file = base_dir / "models" / "scaler.pkl"
onehot_encoder_file = base_dir / "models" / "onehot_encoder.pkl"
ordinal_encoder_file = base_dir / "models" / "ordinal_encoder.pkl"
mlruns_path = base_dir / "mlruns"

# Create all local directories that must exist
required_dirs = [
    raw_data_file.parent,        # "data/raw"
    unprocessed_data_dir,
    preprocessed_data_dir,
    late_model_file.parent,      # "models"
    mlruns_path,
]

for d in required_dirs:
    d.mkdir(parents=True, exist_ok=True)

# S3 bucket paths
S3_MODELS_BASE = "models"
S3_PREPROC_BASE = "preprocessing"


# ─────────────────────────────────────────────
# IMPORT ENV VARS
# ─────────────────────────────────────────────
load_dotenv(dotenv_path=base_dir / ".env")


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
    t_upload_file,
    t_notify,
)


# ─────────────────────────────────────────────
# EXECUTE RETRAIN PIPELINE
# ─────────────────────────────────────────────

@flow
def retrain_pipeline(bucket: str | None = None, region: str | None = None):
    """
    Orchestrate retraining and upload versioned artifacts to S3.
    Uses your existing local paths and uploads individual files with versioned keys.
    """
    logger = get_run_logger()
    
    # Decide once per run (param > env)
    bucket = bucket or os.getenv("S3_BUCKET")
    region = region or os.getenv("AWS_REGION")
    logger.info(f"Config → bucket={bucket}, region={region}")
    
    if not bucket:
        raise ValueError("S3_BUCKET not configured (set env or pass as param).")
    if not region:
        raise ValueError("AWS_REGION not configured (set env or pass as param).")

    # Ensure local directories exist
    late_model_file.parent.mkdir(parents=True, exist_ok=True)
    very_late_model_file.parent.mkdir(parents=True, exist_ok=True)
    unprocessed_data_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
    scaler_file.parent.mkdir(parents=True, exist_ok=True)
    onehot_encoder_file.parent.mkdir(parents=True, exist_ok=True)
    ordinal_encoder_file.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────
    # 1) - 3) Load → Clean → Engineer
    # ─────────────────────────────────────────────
    
    logger.info("- Step 1: Load raw data")
    df = t_load_raw_data.submit(str(raw_data_file)).result()
    logger.info("- Step 2: Clean raw data")
    df = t_clean.submit(df).result()
    logger.info("- Step 3: Feature engineering")
    df = t_engineer.submit(df).result()
    
    # ─────────────────────────────────────────────
    # 4) Preprocess
    # ─────────────────────────────────────────────
    # t_preprocess returns X_train, X_test, y_train, y_test stacked in dictionary
    # t_preprocess also saves scaler/encoders to disk
    logger.info("- Step 4: Preprocess features")
    processed = t_preprocess.submit(
        df,
        True,                   # save_to_disk
        unprocessed_data_dir,   # unprocessed_path
        preprocessed_data_dir,  # preprocessed_path
        scaler_file,            # scaler_file
        onehot_encoder_file,    # onehot_encoder_file
        ordinal_encoder_file,   # ordinal_encoder_file
    ).result()
    
    # ─────────────────────────────────────────────
    # 5) & 6) Train models (tasks return saved model path)
    # ─────────────────────────────────────────────
    logger.info("- Step 5: Train late model")
    late_model_path = t_train_late.submit(
        processed["X_train"], processed["y_late_train"],
        processed["X_test"],  processed["y_late_test"],
        late_model_file,
        mlruns_path
    ).result()

    logger.info("- Step 6: Train very late model")
    very_late_model_path = t_train_very_late.submit(
        processed["X_train"], processed["y_very_late_train"],
        processed["X_test"],  processed["y_very_late_test"],
        very_late_model_file,
        mlruns_path
    ).result()

    # ─────────────────────────────────────────────
    # 7) Upload artifacts to S3
    # ─────────────────────────────────────────────
    # Each run creates a new versioned folder in S3, e.g.:
    #   models/late_model/v2025-10-01/late_model.pkl
    #   preprocessing/v2025-10-01/scaler.pkl
    logger.info("- Step 7: Upload artifacts to S3")
    
    v = datetime.date.today().isoformat()  # e.g. '2025-09-30'
    
    logger.info(f"Uploading artifacts to S3 under version v{v}")
    
    # Upload models (individual upload)
    u1 = t_upload_file.submit(
        local_path=late_model_path,
        bucket=bucket,
        key=f"{S3_MODELS_BASE}/late_model/v{v}/late_model.pkl",
        region=region,
    )
    u2 = t_upload_file.submit(
        local_path=very_late_model_path,
        bucket=bucket,
        key=f"{S3_MODELS_BASE}/very_late_model/v{v}/very_late_model.pkl",
        region=region,
    )

    # Upload preprocessing artifacts (individual upload)
    u3 = t_upload_file.submit(
        local_path=scaler_file,
        bucket=bucket,
        key=f"{S3_PREPROC_BASE}/v{v}/scaler.pkl",
        region=region,
    )
    u4 = t_upload_file.submit(
        local_path=onehot_encoder_file,
        bucket=bucket,
        key=f"{S3_PREPROC_BASE}/v{v}/onehot_encoder.pkl",
        region=region,
    )
    u5 = t_upload_file.submit(
        local_path=ordinal_encoder_file,
        bucket=bucket,
        key=f"{S3_PREPROC_BASE}/v{v}/ordinal_encoder.pkl",
        region=region,
    )
    
    # block until all uploads complete
    for u in (u1, u2, u3, u4, u5):
        u.result()

    t_notify.submit(f"Retrain {v}: models and preprocessing artifacts uploaded.")

if __name__ == "__main__":
    retrain_pipeline()


