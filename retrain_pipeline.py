"""
retrain_pipeline.py

Orchestrated retraining pipeline for late shipment prediction models.

This Prefect flow automates the full lifecycle:
1. Load raw shipment data
2. Clean and validate the data
3. Engineer predictive features
4. Preprocess features (train–test split, save encoder and scaler)
5–6. Train and save the "late" and "very late" Random Forest models
7. (Optional) Upload versioned artifacts to S3 and promote them to the stable 'latest/' keys

Uploads and promotions occur only if UPLOAD_TO_S3=true in the environment.
When disabled, the pipeline still completes locally without requiring AWS access.

Logs metrics to MLflow and sends a Prefect notification on success or failure.
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
    t_overwrite_latest,
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
    
    # Decide whether to upload to S3 (default false)
    upload_to_s3 = os.getenv("UPLOAD_TO_S3", "false").lower() == "true"
    logger.info(f"UPLOAD_TO_S3 = {upload_to_s3}")
    
    if upload_to_s3:
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
    # STEP 7. Upload versioned artifacts → then promote to latest (optional)
    # ─────────────────────────────────────────────
    if upload_to_s3:
        logger.info("- Step 7: Uploading artifacts to S3")
        version_tag = datetime.datetime.now().strftime("v%Y-%m-%d_%H-%M")

        # Each run creates a new versioned folder in S3, e.g.:
        #   models/late_model/v2025-10-01/late_model.pkl
        #   preprocessing/v2025-10-01/scaler.pkl
        #
        # After uploading, each artifact is also copied to its stable 'latest/' path.
        # This overwrites the previous files so the ECS app can always load the newest
        # model and preprocessing artifacts from:
        #   models/late_model/latest/late_model.pkl
        #   preprocessing/latest/scaler.pkl

        # late_model
        u1 = t_upload_file.submit(
            local_path=late_model_file,
            bucket=bucket,
            key=f"{S3_MODELS_BASE}/late_model/{version_tag}/late_model.pkl",
            region=region,
        )
        u1.result()
        t_overwrite_latest.submit(
            versioned_key=f"{S3_MODELS_BASE}/late_model/{version_tag}/late_model.pkl",
            latest_key=f"{S3_MODELS_BASE}/late_model/latest/late_model.pkl",
            bucket=bucket,
            region=region,
        ).result()

        # very_late_model
        u2 = t_upload_file.submit(
            local_path=very_late_model_file,
            bucket=bucket,
            key=f"{S3_MODELS_BASE}/very_late_model/{version_tag}/very_late_model.pkl",
            region=region,
        )
        u2.result()
        t_overwrite_latest.submit(
            versioned_key=f"{S3_MODELS_BASE}/very_late_model/{version_tag}/very_late_model.pkl",
            latest_key=f"{S3_MODELS_BASE}/very_late_model/latest/very_late_model.pkl",
            bucket=bucket,
            region=region,
        ).result()

        # scaler
        u3 = t_upload_file.submit(
            local_path=scaler_file,
            bucket=bucket,
            key=f"{S3_PREPROC_BASE}/{version_tag}/scaler.pkl",
            region=region,
        )
        u3.result()
        t_overwrite_latest.submit(
            versioned_key=f"{S3_PREPROC_BASE}/{version_tag}/scaler.pkl",
            latest_key=f"{S3_PREPROC_BASE}/latest/scaler.pkl",
            bucket=bucket,
            region=region,
        ).result()

        # onehot_encoder
        u4 = t_upload_file.submit(
            local_path=onehot_encoder_file,
            bucket=bucket,
            key=f"{S3_PREPROC_BASE}/{version_tag}/onehot_encoder.pkl",
            region=region,
        )
        u4.result()
        t_overwrite_latest.submit(
            versioned_key=f"{S3_PREPROC_BASE}/{version_tag}/onehot_encoder.pkl",
            latest_key=f"{S3_PREPROC_BASE}/latest/onehot_encoder.pkl",
            bucket=bucket,
            region=region,
        ).result()

        # ordinal_encoder
        u5 = t_upload_file.submit(
            local_path=ordinal_encoder_file,
            bucket=bucket,
            key=f"{S3_PREPROC_BASE}/{version_tag}/ordinal_encoder.pkl",
            region=region,
        )
        u5.result()
        t_overwrite_latest.submit(
            versioned_key=f"{S3_PREPROC_BASE}/{version_tag}/ordinal_encoder.pkl",
            latest_key=f"{S3_PREPROC_BASE}/latest/ordinal_encoder.pkl",
            bucket=bucket,
            region=region,
        ).result()

        t_notify.submit(f"Retrain {version_tag}: uploaded and promoted artifacts to latest/ pointers.")
    else:
        logger.info("UPLOAD_TO_S3 is false → skipping S3 uploads and promotions.")


if __name__ == "__main__":
    retrain_pipeline()


