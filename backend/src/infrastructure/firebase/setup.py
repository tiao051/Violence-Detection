"""Firebase Admin SDK initialization."""

import logging
import os
import firebase_admin
from firebase_admin import credentials
from src.core.config import settings

logger = logging.getLogger(__name__)


def initialize_firebase() -> None:
    """
    Initialize Firebase Admin SDK with Storage bucket.
    
    Idempotent: does nothing if already initialized.
    """
    if firebase_admin._apps:
        return

    try:
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-service-account.json")
        
        if not os.path.exists(cred_path):
            logger.error(f"Firebase credentials file not found at: {cred_path}")
            # Don't raise here, let the app start (auth might fail later but that's ok)
            return

        cred = credentials.Certificate(cred_path)
        
        options = {}
        if settings.firebase_storage_bucket:
            options['storageBucket'] = settings.firebase_storage_bucket
            
        firebase_admin.initialize_app(cred, options)
        logger.info(f"Firebase Admin SDK initialized (Bucket: {settings.firebase_storage_bucket})")
        
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        raise
