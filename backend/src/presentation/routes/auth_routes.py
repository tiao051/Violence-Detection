"""Authentication and authorization routes."""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from pydantic import BaseModel
import jwt
from firebase_admin import auth, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from fastapi import APIRouter, HTTPException, Depends, Header
from src.infrastructure.storage.token_repository import get_token_repository

# Lazy Firestore client initialization
_db = None

def _get_db():
    global _db
    if _db is None:
        try:
            _db = firestore.client()
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            return None
    return _db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# JWT configuration from environment
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = 30

if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY must be set in environment variables")


class FirebaseTokenRequest(BaseModel):
    """Request to verify Firebase ID token."""
    firebase_token: str


class RefreshTokenRequest(BaseModel):
    """Request to refresh access token."""
    refresh_token: str


class FCMTokenRequest(BaseModel):
    """Request to register FCM token for push notifications."""
    token: str
    device_type: str = "android"


class TokenResponse(BaseModel):
    """JWT token response with both access and refresh tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class CameraModel(BaseModel):
    """Camera data model (owned by user)."""
    id: str
    name: str
    location: str
    stream_url: str


def _ensure_firestore_user(user_id: str, email: str):
    """Ensure user exists in Firestore fields."""
    db = _get_db()
    if not db:
        return
    try:
        user_ref = db.collection('users').document(user_id)
        doc = user_ref.get()
        if not doc.exists:
            user_ref.set({
                'uid': user_id,
                'email': email,
                'displayName': email.split('@')[0],
                'authProvider': 'firebase',
                'createdAt': firestore.SERVER_TIMESTAMP,
                'roles': ['user'],
                'disabled': False
            }, merge=True)
            logger.info(f'Auto-created Firestore user document for {user_id}')
    except Exception as e:
        logger.error(f'Error ensuring firestore user: {e}')


def _get_user_owned_cameras(user_id: str) -> list[str]:
    """Get list of camera IDs owned by user from Firestore."""
    db = _get_db()
    if not db:
        return []
    
    try:
        cameras_ref = db.collection('cameras')
        query = cameras_ref.where(filter=FieldFilter("owner_uid", "==", user_id))
        docs = query.stream()
        camera_ids = [doc.id for doc in docs]
        logger.info(f"User {user_id} owns cameras: {camera_ids}")
        return camera_ids
    except Exception as e:
        logger.error(f"Error getting user cameras: {e}")
        return []


def _get_camera_by_id(camera_id: str) -> Optional[Dict[str, Any]]:
    """Get camera details by ID."""
    # TODO: Query Firestore for camera details
    return None


def _create_access_token(user_id: str, email: str, owner_cameras: list[str]) -> tuple[str, int]:
    """Create JWT access token with owner_cameras embedded."""
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + expires_delta
    
    to_encode = {
        "sub": user_id,
        "email": email,
        "owner_cameras": owner_cameras,
        "exp": expire,
        "type": "access"
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, int(expires_delta.total_seconds())


def _create_refresh_token(user_id: str) -> str:
    """Create JWT refresh token."""
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh"
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def _verify_firebase_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify Firebase ID token."""
    try:
        decoded_token = auth.verify_id_token(token, clock_skew_seconds=10)
        return {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "email_verified": decoded_token.get("email_verified", False),
        }
    except Exception as e:
        logger.error(f"Firebase token verification failed: {e}")
        return None


def _extract_user_from_token(authorization: str) -> Optional[Dict[str, Any]]:
    """Extract user info from JWT token in Authorization header."""
    try:
        if not authorization or not authorization.startswith("Bearer "):
            return None
        
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        if payload.get("type") != "access":
            return None
        
        return {
            "uid": payload.get("sub"),
            "email": payload.get("email"),
            "owner_cameras": payload.get("owner_cameras", []),
        }
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        return None
    except Exception as e:
        logger.error(f"Token extraction error: {e}")
        return None


async def get_current_user(authorization: str = Header(None)) -> Dict[str, Any]:
    """Extract current user from JWT token."""
    user = _extract_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token")
    return user


@router.post("/verify-firebase", response_model=TokenResponse)
async def verify_firebase_token(request: FirebaseTokenRequest) -> TokenResponse:
    """Verify Firebase ID token and return JWT tokens."""
    try:
        firebase_claims = _verify_firebase_token(request.firebase_token)
        if not firebase_claims:
            logger.warning("Invalid Firebase token")
            raise HTTPException(status_code=401, detail="Invalid Firebase token")
        
        user_id = firebase_claims.get("uid")
        email = firebase_claims.get("email")
        if not user_id or not email:
            raise HTTPException(status_code=401, detail="Invalid token claims")
        
        # Ensure user exists in Firestore so Admin Dashboard can see them
        _ensure_firestore_user(user_id, email)
        
        owner_cameras = _get_user_owned_cameras(user_id)
        access_token, expires_in = _create_access_token(user_id, email, owner_cameras)
        refresh_token = _create_refresh_token(user_id)
        
        logger.info(f"JWT tokens issued for user {user_id}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            user={
                "uid": user_id,
                "email": email,
                "email_verified": firebase_claims.get("email_verified", False),
                "cameras": owner_cameras,
                "owner_cameras": owner_cameras, 
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying token: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest) -> TokenResponse:
    """Refresh access token using refresh token."""
    try:
        payload = jwt.decode(request.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
            
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
            
        # Get user email and cameras (query Firestore/Auth again)
        try:
            user = auth.get_user(user_id)
            email = user.email
        except Exception:
            raise HTTPException(status_code=401, detail="User not found")
            
        owner_cameras = _get_user_owned_cameras(user_id)
        access_token, expires_in = _create_access_token(user_id, email, owner_cameras)
        new_refresh_token = _create_refresh_token(user_id)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=expires_in,
            user={
                "uid": user_id,
                "email": email,
                "email_verified": user.email_verified,
                "cameras": owner_cameras,
                "owner_cameras": owner_cameras,
            },
        )
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


@router.post("/fcm-token")
async def register_fcm_token(
    request: FCMTokenRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """Register FCM token for push notifications."""
    try:
        user_id = current_user["uid"]
        
        # Save token to Firestore
        db = _get_db()
        if db:
            # Store in user's document or a separate tokens collection
            # Here keeping it simple: users/{uid}/fcm_tokens/{token}
            token_ref = db.collection('users').document(user_id).collection('fcm_tokens').document(request.token)
            token_ref.set({
                "token": request.token,
                "device_type": request.device_type,
                "updated_at": firestore.SERVER_TIMESTAMP
            })
            
            logger.info(f"Registered FCM token for user {user_id}")
            return {"message": "FCM token registered successfully"}
        else:
            raise HTTPException(status_code=500, detail="Database error")
            
    except Exception as e:
        logger.error(f"Error registering FCM token: {e}")
        raise HTTPException(status_code=500, detail="Failed to register token")


# ============ Config copy for Camera Labels ============
CAMERA_LOCATIONS = {
    "cam1": {"name": "Le Trong Tan Intersection", "location": "Tan Phu District"},
    "cam2": {"name": "Cong Hoa Intersection", "location": "Tan Binh District"},
    "cam3": {"name": "Au Co Junction", "location": "Tan Phu District"},
    "cam4": {"name": "Hoa Binh Intersection", "location": "Tan Phu District"},
}

@router.get("/cameras", response_model=list[CameraModel])
async def get_user_cameras(current_user: Dict[str, Any] = Depends(get_current_user)) -> list[CameraModel]:
    """Get list of cameras owned by the authenticated user."""
    try:
        user_id = current_user["uid"]
        owner_cameras = _get_user_owned_cameras(user_id)
        
        cameras = []
        for cam_id in owner_cameras:
            cam_info = CAMERA_LOCATIONS.get(cam_id, {"name": cam_id, "location": "Unknown"})
            cameras.append(CameraModel(
                id=cam_id,
                name=cam_info["name"],
                location=cam_info["location"],
                stream_url=f"/api/v1/streams/{cam_id}/url"
            ))
        
        return cameras
    except Exception as e:
        logger.error(f"Error fetching user cameras: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch cameras")
