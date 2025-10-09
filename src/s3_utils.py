"""
s3_utils.py

Simple helpers for uploading artifacts to Amazon S3.

- Upload one file to a versioned S3 key.
- Promote a versioned object to a stable 'latest/...' key via server-side copy.

Used by retrain_pipeline.py via Prefect to keep both history and a single 'latest' pointer.
"""

import boto3

def upload_file_to_s3(local_path, bucket, key, region=None):
    """
    Upload a local file to s3://{bucket}/{key}.

    Resolves credentials via boto3 (env/profile/role). If `region` is given,
    the client is created for that region. Returns the S3 URI string.
    """

    session = boto3.session.Session(region_name=region)

    # Create an S3 client from the session
    s3 = session.client("s3")

    # Upload the file to the given bucket/key
    s3.upload_file(
        Filename=str(local_path),  # file on your computer
        Bucket=bucket,             # S3 bucket name
        Key=key                    # path/name in S3
    )

    # Return the full "s3://" path for logging or printing
    return f"s3://{bucket}/{key}"


def copy_s3_object(bucket, src_key, dst_key, region=None):
    """
    Copy s3://{bucket}/{src_key} → s3://{bucket}/{dst_key} (server-side).

    Uses S3 CopyObject with MetadataDirective="COPY" so headers/metadata are
    preserved. Returns the destination S3 URI string.
    """
    session = boto3.session.Session(region_name=region)
    s3 = session.client("s3")
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        MetadataDirective="COPY",
    )
    return f"s3://{bucket}/{dst_key}"


def overwrite_latest(bucket, versioned_key, latest_key, region=None):
    """
    Promote a versioned object to the stable 'latest/...' key.

    Wrapper around copy_s3_object; makes latest_key an exact clone of
    versioned_key. Returns the destination S3 URI string.
    """
    return copy_s3_object(bucket=bucket, src_key=versioned_key, dst_key=latest_key, region=region)

