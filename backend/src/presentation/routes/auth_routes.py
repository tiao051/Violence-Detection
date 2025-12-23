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

# Initialize Firestore client for camera queries
try:
    _db = firestore.client()
except Exception:
    _db = None

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


def _get_user_owned_cameras(user_id: str) -> list[str]:
    """Get list of camera IDs owned by user from Firestore."""
    if not _db:
        return []
    
    try:
        cameras_ref = _db.collection('cameras')
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
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, int(expires_delta.total_seconds())


def _create_refresh_token(user_id: str) -> str:
    """Create JWT refresh token (30 days)."""
    expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    expire = datetime.now(timezone.utc) + expires_delta
    
    to_encode = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh",
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
                "owner_cameras": owner_cameras,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(status_code=500, detail="Token verification failed")


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(request: RefreshTokenRequest) -> TokenResponse:
    """Refresh access token using refresh token."""
    try:
        payload = jwt.decode(request.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user ID in token")
        
        owner_cameras = _get_user_owned_cameras(user_id)
        access_token, expires_in = _create_access_token(user_id, "", owner_cameras)
        
        logger.info(f"Access token refreshed for user {user_id}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=request.refresh_token,
            expires_in=expires_in,
            user={
                "uid": user_id,
                "owner_cameras": owner_cameras,
            },
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


@router.get("/cameras", response_model=list[CameraModel])
async def get_user_cameras(current_user: Dict[str, Any] = Depends(get_current_user)) -> list[CameraModel]:
    """Get list of cameras owned by current user."""
    try:
        owner_cameras = current_user.get("owner_cameras", [])
        if not owner_cameras:
            logger.info(f"User {current_user.get('uid')} owns no cameras")
            return []
        
        cameras = []
        for camera_id in owner_cameras:
            camera = _get_camera_by_id(camera_id)
            if camera:
                cameras.append(
                    CameraModel(
                        id=camera["id"],
                        name=camera["name"],
                        location=camera["location"],
                        stream_url=camera["stream_url"],
                    )
                )
        
        logger.info(f"Retrieved {len(cameras)} cameras for user {current_user.get('uid')}")
        return cameras
    except Exception as e:
        logger.error(f"Error retrieving cameras: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cameras")


@router.get("/streams/{camera_id}/url")
async def get_stream_url(
    camera_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get authorized stream URL for camera."""
    try:
        user_id = current_user.get("uid")
        owner_cameras = current_user.get("owner_cameras", [])
        
        if camera_id not in owner_cameras:
            logger.warning(f"User {user_id} attempted unauthorized access to camera {camera_id}")
            raise HTTPException(status_code=403, detail="Not authorized to access this camera")
        
        camera = _get_camera_by_id(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        logger.info(f"Stream URL provided to user {user_id} for camera {camera_id}")
        
        return {
            "camera_id": camera_id,
            "stream_url": camera["stream_url"],
            "type": "webrtc",
            "expires_at": expires_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stream URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stream URL")


@router.post("/register-fcm-token")
async def register_fcm_token(
    request: FCMTokenRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, str]:
    """Register FCM token for push notifications."""
    try:
        user_id = current_user.get("uid")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user")
        
        token_repo = get_token_repository()
        success = token_repo.save_token(
            user_id=user_id,
            token=request.token,
            device_type=request.device_type
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to register FCM token")
        
        logger.info(f"FCM token registered for user {user_id} (device: {request.device_type})")
        return {
            "message": "FCM token registered successfully",
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering FCM token: {e}")
        raise HTTPException(status_code=500, detail="Failed to register FCM token")
