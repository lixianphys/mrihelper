"""
S3 Storage Utility for MRI Master
Handles file operations with S3-compatible storage (MinIO or AWS S3)
"""
import os
import logging
import boto3
from botocore.exceptions import ClientError
from typing import List, BinaryIO

## Comment this line to avoid loading .env file in production environment, which should not contain any secrets.
# from dotenv import load_dotenv
# load_dotenv(dotenv_path=os.getenv("ENV_FILE"))


# --- Environment Variables ---

S3_UPLOAD_PREFIX = os.getenv("S3_UPLOAD_PREFIX")
S3_OUTPUT_PREFIX = os.getenv("S3_OUTPUT_PREFIX")
USE_S3_STORAGE = os.getenv("USE_S3_STORAGE")
S3_PROVIDER = os.getenv("S3_PROVIDER")  
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION")

logger = logging.getLogger("mrimaster-s3storage")

class S3Storage:
    """Handles all S3 storage operations for the application"""
    
    def __init__(self):
        """Initialize S3 client and ensure the bucket exists"""
        # Configure boto3 client based on provider
        config_kwargs = {
            'aws_access_key_id': S3_ACCESS_KEY,
            'aws_secret_access_key': S3_SECRET_KEY,
            'region_name': S3_REGION,
        }
        
        # For MinIO or other S3-compatible services, add endpoint_url
        if S3_PROVIDER == 'minio':
            config_kwargs['endpoint_url'] = S3_ENDPOINT
            config_kwargs['config'] = boto3.session.Config(signature_version='s3v4')
            logger.info(f"Using MinIO storage at endpoint: {S3_ENDPOINT}")
        else:
            logger.info(f"Using AWS S3 storage in region: {S3_REGION}")
        
        self.s3_client = boto3.client('s3', **config_kwargs)
        self.bucket_name = S3_BUCKET_NAME
        self._ensure_bucket_exists()
        
    def _ensure_bucket_exists(self):
        """Create the bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket '{self.bucket_name}' exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket does not exist, create it
                try:
                    if S3_PROVIDER == 'minio':
                        # For MinIO, simple bucket creation
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        # For AWS S3, handle region-specific bucket creation
                        if S3_REGION == 'us-east-1':
                            # us-east-1 doesn't use LocationConstraint
                            self.s3_client.create_bucket(Bucket=self.bucket_name)
                        else:
                            self.s3_client.create_bucket(
                                Bucket=self.bucket_name,
                                CreateBucketConfiguration={'LocationConstraint': S3_REGION}
                            )
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                except ClientError as create_err:
                    logger.error(f"Failed to create bucket: {str(create_err)}")
                    raise
            else:
                logger.error(f"Error checking S3 bucket: {str(e)}")
                raise

    def save_file(self, file_data: BinaryIO, s3_key: str) -> str:
        """
        Save file to S3 storage
        Returns: the S3 key of the saved file
        """
        try:
            self.s3_client.upload_fileobj(file_data, self.bucket_name, s3_key)
            logger.info(f"Successfully saved file to S3: {s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Error saving file to S3: {str(e)}")
            raise
            
    def save_bytes(self, data: bytes, s3_key: str) -> str:
        """
        Save bytes data to S3 storage
        Returns: the S3 key of the saved file
        """
        try:
            self.s3_client.put_object(Body=data, Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Successfully saved data to S3: {s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Error saving data to S3: {str(e)}")
            raise
    
    def load_file(self, s3_key: str) -> bytes:
        """
        Load file from S3 storage
        Returns: file contents as bytes
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"Error loading file from S3: {str(e)}")
            raise
            
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3
        Returns: True if successful, False if error
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted S3 file: {s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting S3 file {s3_key}: {str(e)}")
            return False
            
    def list_files(self, prefix: str) -> List[str]:
        """
        List files in S3 with given prefix
        Returns: List of S3 keys
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            keys = []
            if 'Contents' in response:
                keys = [item['Key'] for item in response['Contents']]
            return keys
        except ClientError as e:
            logger.error(f"Error listing S3 files with prefix {prefix}: {str(e)}")
            return []
            
    def get_download_url(self, s3_key: str, expires_in=3600) -> str:
        """
        Generate a presigned URL for downloading a file
        Returns: Presigned URL string
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL for {s3_key}: {str(e)}")
            raise

# Create a singleton instance for the application to use
s3_storage = S3Storage() if USE_S3_STORAGE == 'true' else None
