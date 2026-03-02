"""S3/MinIO file storage backend for LiteAgent."""

import logging
import mimetypes
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class StorageBackend:
    """S3-compatible (MinIO) file storage."""

    def __init__(self, config: dict):
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            raise ImportError("boto3 is required: pip install liteagent[storage]")

        self.bucket = config.get("bucket", "liteagent")
        endpoint = config.get("endpoint", "http://localhost:9000")

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=config.get("access_key", ""),
            aws_secret_access_key=config.get("secret_key", ""),
            region_name=config.get("region", "us-east-1"),
            config=BotoConfig(signature_version="s3v4"),
        )
        self._endpoint = endpoint
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except Exception:
            try:
                self._client.create_bucket(Bucket=self.bucket)
                logger.info("Created bucket: %s", self.bucket)
            except Exception as e:
                logger.warning("Could not create bucket %s: %s", self.bucket, e)

    def upload(self, key: str, data: bytes, content_type: str = "") -> str:
        """Upload bytes to storage. Returns the key."""
        if not content_type:
            content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
        self._client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        logger.debug("Uploaded %s (%d bytes)", key, len(data))
        return key

    def upload_file(self, local_path: str, key: str = None) -> str:
        """Upload a local file. If key is None, uses the filename."""
        p = Path(local_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        if key is None:
            key = p.name
        content_type = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        self._client.upload_file(
            str(p), self.bucket, key, ExtraArgs={"ContentType": content_type})
        logger.debug("Uploaded file %s → %s", p.name, key)
        return key

    def download(self, key: str) -> bytes:
        """Download file content by key."""
        resp = self._client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def download_to(self, key: str, local_path: str):
        """Download file to a local path."""
        self._client.download_file(self.bucket, key, local_path)

    def delete(self, key: str) -> bool:
        """Delete a file. Returns True on success."""
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as e:
            logger.warning("Delete failed for %s: %s", key, e)
            return False

    def list_files(self, prefix: str = "", limit: int = 100) -> list[dict]:
        """List files with metadata."""
        kwargs = {"Bucket": self.bucket, "MaxKeys": limit}
        if prefix:
            kwargs["Prefix"] = prefix
        resp = self._client.list_objects_v2(**kwargs)
        files = []
        for obj in resp.get("Contents", []):
            files.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat()
                if isinstance(obj["LastModified"], datetime) else str(obj["LastModified"]),
            })
        return files

    def exists(self, key: str) -> bool:
        """Check if a file exists."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def get_url(self, key: str, expires: int = 3600) -> str:
        """Generate a presigned URL for direct access."""
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )

    def get_stats(self) -> dict:
        """Get storage stats: file count and total size."""
        resp = self._client.list_objects_v2(Bucket=self.bucket)
        contents = resp.get("Contents", [])
        total_size = sum(obj["Size"] for obj in contents)
        return {"file_count": len(contents), "total_size_bytes": total_size}


def create_storage(config: dict) -> StorageBackend | None:
    """Create StorageBackend from config, or None if disabled."""
    storage_cfg = config.get("storage", {})
    if not storage_cfg.get("enabled", False):
        return None

    # Try loading credentials from keys.json
    from .config import get_api_key
    access_key = storage_cfg.get("access_key") or get_api_key("minio_access") or ""
    secret_key = storage_cfg.get("secret_key") or get_api_key("minio_secret") or ""

    merged = {
        "endpoint": storage_cfg.get("endpoint", "http://localhost:9000"),
        "access_key": access_key,
        "secret_key": secret_key,
        "bucket": storage_cfg.get("bucket", "liteagent"),
        "region": storage_cfg.get("region", "us-east-1"),
    }

    try:
        return StorageBackend(merged)
    except Exception as e:
        logger.warning("Storage backend initialization failed: %s", e)
        return None
