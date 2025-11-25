"""Authentication and authorization routes for mobile app (Flutter).

Owner-only model: Each user owns cameras. JWT contains owner_cameras list.
No sharing - each user only sees their own cameras.
"""

import logging
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from pydantic import BaseModel
import jwt
import firebase_admin
from firebase_admin import credentials, auth

from fastapi import APIRouter, HTTPException, Depends, Header

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# JWT configuration from environment
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = 30

if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY must be set in environment variables")

# Firebase Admin SDK initialization
firebase_credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-service-account.json")
if not firebase_admin._apps:  # Only initialize if not already initialized
    try:
        cred = credentials.Certificate(firebase_credentials_path)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
        raise


class FirebaseTokenRequest(BaseModel):
    """Request to verify Firebase ID token."""
    firebase_token: str


class RefreshTokenRequest(BaseModel):
    """Request to refresh access token."""
    refresh_token: str


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
    """
    Get list of camera IDs owned by user.
    
    In production, query Firestore:
    db.collection('cameras').where('owner_uid', '==', user_id).get()
    
    Args:
        user_id: Firebase user ID
        
    Returns:
        List of camera IDs user owns
    """
    # TODO: Query Firestore for cameras where owner_uid == user_id
    # For now, mock data
    
    mock_camera_owners = {
        "user_123": ["cam1", "cam2", "cam3", "cam4"],
        "user_456": ["cam1", "cam2", "cam3", "cam4"],
        
        "H2399Gybu8TeslP8zyEyP4uhn0l2": ["cam1", "cam2", "cam3", "cam4"],
    }
    
    return mock_camera_owners.get(user_id, [])


def _get_camera_by_id(camera_id: str) -> Optional[Dict[str, Any]]:
    """
    Get camera details by ID.
    
    In production, query Firestore:
    db.collection('cameras').document(camera_id).get()
    
    Args:
        camera_id: Camera ID
        
    Returns:
        Camera object or None if not found
    """
    # TODO: Query Firestore for camera details
    # Mock data
    mock_cameras = {
        "cam1": {
            "id": "cam1",
            "name": "Front Gate",
            "location": "Entrance",
            "stream_url": "rtsp://localhost:8554/cam1",
            "owner_uid": "user_123",
        },
        "cam2": {
            "id": "cam2",
            "name": "Back Yard",
            "location": "Rear",
            "stream_url": "rtsp://localhost:8554/cam2",
            "owner_uid": "user_123",
        },
        "cam3": {
            "id": "cam3",
            "name": "Front Door",
            "location": "Main Door",
            "stream_url": "rtsp://localhost:8554/cam3",
            "owner_uid": "user_123",
        },
        "cam4": {
            "id": "cam4",
            "name": "Living Room",
            "location": "Living Room",
            "stream_url": "rtsp://localhost:8554/cam4",
            "owner_uid": "user_456",
        },
        "cam5": {
            "id": "cam5",
            "name": "Garage",
            "location": "Garage",
            "stream_url": "rtsp://localhost:8554/cam5",
            "owner_uid": "user_456",
        },
    }
    
    return mock_cameras.get(camera_id)


def _create_access_token(user_id: str, email: str, owner_cameras: list[str]) -> tuple[str, int]:
    """
    Create JWT access token with owner_cameras embedded.
    
    Key insight: JWT contains owner_cameras list so backend doesn't need database queries.
    
    Args:
        user_id: Firebase user ID
        email: User email
        owner_cameras: List of camera IDs user owns
        
    Returns:
        Tuple of (token, expires_in_seconds)
    """
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + expires_delta
    
    to_encode = {
        "sub": user_id,                    # Subject (user ID)
        "email": email,
        "owner_cameras": owner_cameras,    # KEY: Cameras user owns - embedded in JWT
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, int(expires_delta.total_seconds())


def _create_refresh_token(user_id: str) -> str:
    """
    Create JWT refresh token (long-lived, 30 days).
    
    Args:
        user_id: Firebase user ID
        
    Returns:
        Refresh token
    """
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
    """
    Verify Firebase ID token using Firebase Admin SDK.
    
    Args:
        token: Firebase ID token
        
    Returns:
        Decoded token claims if valid, None otherwise
    """
    try:
        # Verify with Firebase Admin SDK
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
    """
    Extract user info from JWT token in Authorization header.
    
    Args:
        authorization: Authorization header value (e.g., "Bearer token")
        
    Returns:
        Decoded token claims if valid, None otherwise
    """
    try:
        if not authorization or not authorization.startswith("Bearer "):
            return None
        
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Verify token type is 'access'
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
    """
    Dependency to extract current user from JWT token.
    
    Args:
        authorization: Authorization header
        
    Returns:
        User info from token (uid, email, owner_cameras)
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    user = _extract_user_from_token(authorization)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token")
    
    return user


@router.post("/verify-firebase", response_model=TokenResponse)
async def verify_firebase_token(request: FirebaseTokenRequest) -> TokenResponse:
    """
    Verify Firebase ID token and return JWT access + refresh tokens.
    
    This endpoint exchanges a Firebase ID token for backend JWT tokens.
    The JWT access token contains owner_cameras list for zero-database queries.
    
    Args:
        request: Firebase token request
        
    Returns:
        Access token, refresh token, and user info
        
    Raises:
        HTTPException: If token verification fails
    """
    try:
        # Verify Firebase token
        firebase_claims = _verify_firebase_token(request.firebase_token)
        
        if not firebase_claims:
            logger.warning("Invalid Firebase token")
            raise HTTPException(status_code=401, detail="Invalid Firebase token")
        
        user_id = firebase_claims.get("uid")
        email = firebase_claims.get("email")
        
        if not user_id or not email:
            raise HTTPException(status_code=401, detail="Invalid token claims")
        
        # Get cameras owned by user
        owner_cameras = _get_user_owned_cameras(user_id)
        
        # Create JWT tokens
        access_token, expires_in = _create_access_token(user_id, email, owner_cameras)
        refresh_token = _create_refresh_token(user_id)
        
        logger.info(f"JWT tokens issued for user {user_id} with cameras: {owner_cameras}")
        
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
    """
    Refresh access token using refresh token.
    
    Args:
        request: Refresh token request
        
    Returns:
        New access token with updated expiry
        
    Raises:
        HTTPException: If refresh token is invalid
    """
    try:
        # Verify refresh token
        payload = jwt.decode(request.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user ID in token")
        
        # Get current camera ownership
        owner_cameras = _get_user_owned_cameras(user_id)
        
        # Create new access token
        # Note: We don't have email in refresh token, but that's OK for access token
        access_token, expires_in = _create_access_token(user_id, "", owner_cameras)
        
        logger.info(f"Access token refreshed for user {user_id}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=request.refresh_token,  # Refresh token stays the same
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
    """
    Get list of cameras owned by current user.
    
    NO DATABASE QUERY NEEDED - owner_cameras list is in JWT token!
    
    Args:
        current_user: Current user (from JWT token)
        
    Returns:
        List of cameras owned by user
    """
    try:
        owner_cameras = current_user.get("owner_cameras", [])
        
        if not owner_cameras:
            logger.info(f"User {current_user.get('uid')} owns no cameras")
            return []
        
        # Build camera list from owner_cameras
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
    """
    Get authorized stream URL for camera.
    
    Validates that user owns the camera (check owner_cameras from JWT).
    
    Args:
        camera_id: Camera ID (e.g., "cam1")
        current_user: Current user (from JWT token)
        
    Returns:
        Stream URL and metadata
        
    Raises:
        HTTPException: If user doesn't own camera or camera not found
    """
    try:
        user_id = current_user.get("uid")
        owner_cameras = current_user.get("owner_cameras", [])
        
        # Check if user owns this camera (from JWT)
        if camera_id not in owner_cameras:
            logger.warning(f"User {user_id} attempted unauthorized access to camera {camera_id}")
            raise HTTPException(status_code=403, detail="Not authorized to access this camera")
        
        # Get camera details
        camera = _get_camera_by_id(camera_id)
        
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        # Calculate expiration time
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        
        logger.info(f"Stream URL provided to user {user_id} for camera {camera_id}")
        
        return {
            "camera_id": camera_id,
            "stream_url": camera["stream_url"],
            "type": "webrtc",  # WebRTC via WHEP
            "expires_at": expires_at,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stream URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stream URL")
