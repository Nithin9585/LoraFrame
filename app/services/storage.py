"""
Storage Service
Handles file storage - supports Google Cloud Storage, S3, and local filesystem.
"""

import os
import uuid
from pathlib import Path
from fastapi import UploadFile
from app.core.config import settings


class StorageService:
    """Service for file storage operations."""
    
    def __init__(self):
        # Priority: GCS > S3 > Local
        self.use_gcs = settings.USE_GCS
        self.use_local = settings.USE_LOCAL_STORAGE and not self.use_gcs
        
        if self.use_gcs:
            # Google Cloud Storage
            from google.cloud import storage
            self.gcs_client = storage.Client(project=settings.GCP_PROJECT_ID)
            self.bucket_uploads = self.gcs_client.bucket(settings.GCS_BUCKET_UPLOADS)
            self.bucket_outputs = self.gcs_client.bucket(settings.GCS_BUCKET_OUTPUTS)
            print(f"[Storage] Using Google Cloud Storage: {settings.GCS_BUCKET_UPLOADS}, {settings.GCS_BUCKET_OUTPUTS}")
            
        elif self.use_local:
            self.base_path = Path(settings.LOCAL_STORAGE_PATH)
            self.base_path.mkdir(parents=True, exist_ok=True)
            print(f"[Storage] Using local storage: {self.base_path}")
            
        else:
            # S3 fallback
            import boto3
            from botocore.config import Config
            self.s3 = boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
                region_name=settings.S3_REGION,
                config=Config(signature_version="s3v4")
            )
            self.bucket = settings.S3_BUCKET
            print(f"[Storage] Using S3: {self.bucket}")
    
    async def upload_file(self, file: UploadFile, path: str) -> str:
        """Upload file and return URL."""
        content = await file.read()
        return await self.upload_bytes(content, path, file.content_type or "image/jpeg")
    
    async def upload_bytes(self, data: bytes, path: str, content_type: str = "image/jpeg") -> str:
        """Upload bytes and return URL."""
        if self.use_gcs:
            return await self._upload_gcs(data, path, content_type)
        elif self.use_local:
            return await self._upload_local(data, path)
        else:
            return await self._upload_s3(data, path, content_type)
    
    async def _upload_gcs(self, data: bytes, path: str, content_type: str) -> str:
        """Upload to Google Cloud Storage."""
        # Determine bucket based on path
        if path.startswith("uploads/") or path.startswith("characters/"):
            bucket = self.bucket_uploads
        else:
            bucket = self.bucket_outputs
        
        blob = bucket.blob(path)
        blob.upload_from_string(data, content_type=content_type)
        
        # Return API URL that proxies the file
        # This allows frontend to access files without GCS permissions
        return f"/files/{path}"
    
    async def _upload_local(self, data: bytes, path: str) -> str:
        """Save file to local filesystem."""
        file_path = self.base_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(data)
        
        # Return local file URL
        return f"file://{file_path.absolute()}"
    
    async def _upload_s3(self, data: bytes, path: str, content_type: str) -> str:
        """Upload to S3."""
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=path, 
            Body=data, 
            ContentType=content_type
        )
        return f"s3://{self.bucket}/{path}"
    
    async def delete_file(self, path: str):
        """Delete a single file."""
        if self.use_gcs:
            # Delete from GCS - determine correct bucket
            if path.startswith("uploads/") or path.startswith("characters/"):
                bucket = self.bucket_uploads
            else:
                bucket = self.bucket_outputs
            blob = bucket.blob(path)
            try:
                blob.delete()
                print(f"[Storage] Deleted file: {path}")
            except Exception as e:
                print(f"[Storage] Warning: Could not delete {path}: {e}")
        elif self.use_local:
            file_path = self.base_path / path
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                print(f"[Storage] Deleted file: {path}")
        else:
            # S3
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=path)
                print(f"[Storage] Deleted file: {path}")
            except Exception as e:
                print(f"[Storage] Warning: Could not delete {path}: {e}")
    
    async def delete_folder(self, prefix: str):
        """Delete all files with given prefix."""
        if self.use_gcs:
            # Delete from GCS - character images and uploads from uploads bucket
            if prefix.startswith("uploads/") or prefix.startswith("characters/"):
                bucket = self.bucket_uploads
            else:
                bucket = self.bucket_outputs
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                blob.delete()
        elif self.use_local:
            folder_path = self.base_path / prefix
            if folder_path.exists():
                import shutil
                shutil.rmtree(folder_path)
        else:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            if "Contents" in response:
                objects = [{"Key": obj["Key"]} for obj in response["Contents"]]
                self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
    
    async def get_file(self, path: str) -> bytes:
        """Get file contents."""
        if self.use_gcs:
            # Character images and uploads go to uploads bucket
            # Everything else goes to outputs bucket
            if path.startswith("uploads/") or path.startswith("characters/"):
                bucket = self.bucket_uploads
            else:
                bucket = self.bucket_outputs
            blob = bucket.blob(path)
            return blob.download_as_bytes()
        elif self.use_local:
            file_path = self.base_path / path
            with open(file_path, "rb") as f:
                return f.read()
        else:
            response = self.s3.get_object(Bucket=self.bucket, Key=path)
            return response["Body"].read()
    
    def get_public_url(self, path: str) -> str:
        """Get public URL for file."""
        # Return API proxy URL for all storage backends
        # This allows consistent access through the API endpoint
        return f"/files/{path}"
    
    def convert_gcs_url_to_api(self, url: str) -> str:
        """Convert old GCS URL to new API proxy URL."""
        if url.startswith("https://storage.googleapis.com/"):
            # Extract path from GCS URL
            # Format: https://storage.googleapis.com/bucket-name/path/to/file.jpg
            parts = url.replace("https://storage.googleapis.com/", "").split("/", 1)
            if len(parts) > 1:
                path = parts[1]  # Get everything after bucket name
                return f"/files/{path}"
        # Return as-is if not a GCS URL
        return url
    
    async def download_bytes(self, url: str) -> bytes:
        """
        Download file bytes from a URL (GCS, local, S3, HTTP, or API proxy).
        
        Args:
            url: File URL (file://, s3://, /files/, https://storage.googleapis.com/, or http(s)://)
            
        Returns:
            File bytes
        """
        try:
            # Handle API proxy URLs (e.g., /files/characters/...)
            if url.startswith("/files/"):
                # Strip /files/ prefix and use as path
                path = url.replace("/files/", "", 1)
                return await self.get_file(path)
            
            # Handle old GCS URLs - convert to path and fetch directly
            elif url.startswith("https://storage.googleapis.com/"):
                # Extract path from GCS URL
                parts = url.replace("https://storage.googleapis.com/", "").split("/", 1)
                if len(parts) > 1:
                    path = parts[1]  # Everything after bucket name
                    print(f"[Storage] Converting old GCS URL to path: {path}")
                    return await self.get_file(path)
                else:
                    raise ValueError(f"Invalid GCS URL format: {url}")
            
            elif url.startswith("file://"):
                # Local file
                file_path = url.replace("file://", "")
                with open(file_path, "rb") as f:
                    return f.read()
            
            elif url.startswith("s3://"):
                # S3 path
                parts = url.replace("s3://", "").split("/", 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ""
                response = self.s3.get_object(Bucket=bucket, Key=key)
                return response["Body"].read()
            
            elif url.startswith(("http://", "https://")):
                # HTTP URL (external)
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=60.0)
                    response.raise_for_status()
                    return response.content
            
            else:
                # Assume it's a relative path
                return await self.get_file(url)
        except Exception as e:
            print(f"[Storage] Error downloading {url}: {e}")
            raise

