"""
src/tasks.py

Prefect task wrappers around existing pipeline functions
and S3 helper functions.
"""

import os
from prefect import task, get_run_logger

# your existing pipeline functions
from src.load_data import load_raw_data
from src.clean_data import clean_raw_data
from src.feature_engineering import engineer_features
from src.preprocess_features import preprocess_features
from src.train_late_model import train_late_model
from src.train_very_late_model import train_very_late_model

from src.s3_utils import (
    upload_file_to_s3,
    copy_s3_object,
    overwrite_latest,
)



# ─────────────────────────────────────────────
# Wrap core pipeline functions in tasks
# ─────────────────────────────────────────────

@task(retries=2, retry_delay_seconds=60)
def t_load_raw_data(raw_csv_path):
    logger = get_run_logger()
    logger.info(f"Loading raw data from {raw_csv_path}")
    return load_raw_data(raw_csv_path)

@task(retries=2, retry_delay_seconds=60)
def t_clean(df):
    return clean_raw_data(df)

@task(retries=2, retry_delay_seconds=60)
def t_engineer(df):
    return engineer_features(df)

@task(retries=2, retry_delay_seconds=60)
def t_preprocess(
        df, 
        save_to_disk, 
        unprocessed_path, 
        preprocessed_path,
        scaler_file,
        onehot_encoder_file,
        ordinal_encoder_file
    ):
    return preprocess_features(
        df=df,
        save_to_disk=save_to_disk,
        unprocessed_path=unprocessed_path,
        preprocessed_path=preprocessed_path,
        scaler_file=scaler_file,
        onehot_encoder_file=onehot_encoder_file,
        ordinal_encoder_file=ordinal_encoder_file,
    )

@task(retries=2, retry_delay_seconds=60)
def t_train_late(X_train, y_train, X_test, y_test, model_file, mlruns_path):
    logger = get_run_logger()
    logger.info(f"Training late model → {model_file}")
    train_late_model(X_train, y_train, X_test, y_test, model_file, mlruns_path)
    return model_file

@task(retries=2, retry_delay_seconds=60)
def t_train_very_late(X_train, y_train, X_test, y_test, model_file, mlruns_path):
    logger = get_run_logger()
    logger.info(f"Training very late model → {model_file}")
    train_very_late_model(X_train, y_train, X_test, y_test, model_file, mlruns_path)
    return model_file


# ─────────────────────────────────────────────
# Wrap S3 upload helper functions in tasks
# ─────────────────────────────────────────────

@task(retries=2, retry_delay_seconds=30)
def t_upload_file(local_path, bucket=None, key=None, region=None):
    """
    Upload a single file to S3. Return s3://... URI.
    bucket/region default to env if not provided.
    """
    logger = get_run_logger()
    bucket = bucket or os.getenv("S3_BUCKET")
    region = region or os.getenv("AWS_REGION")
    if not bucket:
        raise ValueError("S3_BUCKET not set (pass bucket=... or set env var)")
    if not key:
        raise ValueError("S3 key must be provided") 
    uri = upload_file_to_s3(local_path, bucket=bucket, key=key, region=region)
    logger.info(f"Uploaded file → {uri}")
    return uri


@task(retries=2, retry_delay_seconds=15)
def t_copy_s3(src_key, dst_key, bucket=None, region=None):
    """
    Server-side copy: s3://{bucket}/{src_key} → s3://{bucket}/{dst_key}.
    """
    logger = get_run_logger()
    bucket = bucket or os.getenv("ARTIFACT_BUCKET") or os.getenv("S3_BUCKET")
    region = region or os.getenv("AWS_REGION")
    if not bucket:
        raise ValueError("ARTIFACT_BUCKET (or S3_BUCKET) not set; pass bucket=... or set env var")
    uri = copy_s3_object(bucket=bucket, src_key=src_key, dst_key=dst_key, region=region)
    logger.info(f"Copied object → {uri}")
    return uri

@task(retries=2, retry_delay_seconds=15)
def t_overwrite_latest(versioned_key, latest_key, bucket=None, region=None):
    """
    Promote versioned → latest (exact clone) in the same bucket.
    """
    logger = get_run_logger()
    bucket = bucket or os.getenv("ARTIFACT_BUCKET") or os.getenv("S3_BUCKET")
    region = region or os.getenv("AWS_REGION")
    if not bucket:
        raise ValueError("ARTIFACT_BUCKET (or S3_BUCKET) not set; pass bucket=... or set env var")
    uri = overwrite_latest(bucket=bucket, versioned_key=versioned_key, latest_key=latest_key, region=region)
    logger.info(f"Promoted to latest → {uri}")
    return uri

# ─────────────────────────────────────────────
# Notification task (logs messages for Prefect)
# ─────────────────────────────────────────────
@task
def t_notify(message: str):
    logger = get_run_logger()
    logger.info(message)
