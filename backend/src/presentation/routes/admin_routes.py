"""Admin routes for user and camera management."""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from firebase_admin import firestore, auth
from google.cloud.firestore_v1.base_query import FieldFilter

from fastapi import APIRouter, HTTPException
from src.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

# Lazy Firestore client initialization
_db = None


def _get_db():
    """Get Firestore client (lazy initialization)."""
    global _db
    if _db is None:
        try:
            _db = firestore.client()
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            return None
    return _db


# ============ Pydantic Models ============

class UserResponse(BaseModel):
    """User response model."""
    uid: str
    email: str
    displayName: Optional[str] = None
    authProvider: Optional[str] = None
    createdAt: Optional[str] = None
    photoUrl: Optional[str] = None
    disabled: bool = False
    camerasCount: int = 0


class UserDetailResponse(UserResponse):
    """User detail with assigned cameras."""
    cameras: List[Dict[str, Any]] = []


class CameraResponse(BaseModel):
    """Camera response model."""
    id: str
    name: str
    location: str
    owner_uid: Optional[str] = None
    owner_email: Optional[str] = None


class AssignCameraRequest(BaseModel):
    """Request to assign camera to user."""
    user_uid: Optional[str] = None  # None = unassign


class UpdateUserStatusRequest(BaseModel):
    """Request to enable/disable user."""
    disabled: bool


# ============ Camera Location Data (from existing hardcode) ============

CAMERA_LOCATIONS = {
    "cam1": {"name": "Le Trong Tan Intersection", "location": "Tan Phu District"},
    "cam2": {"name": "Cong Hoa Intersection", "location": "Tan Binh District"},
    "cam3": {"name": "Au Co Junction", "location": "Tan Phu District"},
    "cam4": {"name": "Hoa Binh Intersection", "location": "Tan Phu District"},
}


# ============ Helper Functions ============

def _get_cameras_for_user(user_uid: str) -> List[Dict[str, Any]]:
    """Get all cameras assigned to a user."""
    db = _get_db()
    if not db:
        return []
    
    try:
        cameras_ref = db.collection('cameras')
        query = cameras_ref.where(filter=FieldFilter("owner_uid", "==", user_uid))
        docs = query.stream()
        
        cameras = []
        for doc in docs:
            cam_info = CAMERA_LOCATIONS.get(doc.id, {"name": doc.id, "location": "Unknown"})
            cameras.append({
                "id": doc.id,
                "name": cam_info["name"],
                "location": cam_info["location"],
            })
        return cameras
    except Exception as e:
        logger.error(f"Error getting cameras for user {user_uid}: {e}")
        return []


def _get_user_email(user_uid: str) -> Optional[str]:
    """Get user email from Firestore."""
    db = _get_db()
    if not db or not user_uid:
        return None
    
    try:
        user_doc = db.collection('users').document(user_uid).get()
        if user_doc.exists:
            return user_doc.to_dict().get("email")
        return None
    except Exception as e:
        logger.error(f"Error getting user email: {e}")
        return None


def _count_cameras_for_user(user_uid: str) -> int:
    """Count cameras assigned to a user."""
    db = _get_db()
    if not db:
        return 0
    
    try:
        cameras_ref = db.collection('cameras')
        query = cameras_ref.where(filter=FieldFilter("owner_uid", "==", user_uid))
        docs = query.stream()
        return sum(1 for _ in docs)
    except Exception as e:
        logger.error(f"Error counting cameras: {e}")
        return 0


def _get_camera_owner(camera_id: str) -> Optional[str]:
    """Get owner_uid for a camera from Firestore."""
    db = _get_db()
    if not db:
        return None
    
    try:
        doc = db.collection('cameras').document(camera_id).get()
        if doc.exists:
            return doc.to_dict().get("owner_uid")
        return None
    except Exception as e:
        logger.error(f"Error getting camera owner: {e}")
        return None


# ============ User Endpoints ============

@router.get("/users", response_model=List[UserResponse])
async def list_users() -> List[UserResponse]:
    """Get all users from Firestore."""
    db = _get_db()
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        users_ref = db.collection('users')
        docs = users_ref.stream()
        
        users = []
        for doc in docs:
            data = doc.to_dict()
            uid = doc.id
            
            # Check if user is disabled in Firebase Auth
            disabled = False
            try:
                firebase_user = auth.get_user(uid)
                disabled = firebase_user.disabled
            except Exception:
                pass  # User might not exist in Auth
            
            users.append(UserResponse(
                uid=uid,
                email=data.get("email", ""),
                displayName=data.get("displayName"),
                authProvider=data.get("authProvider"),
                createdAt=str(data.get("createdAt")) if data.get("createdAt") else None,
                photoUrl=data.get("photoUrl"),
                disabled=disabled,
                camerasCount=_count_cameras_for_user(uid)
            ))
        
        logger.info(f"Retrieved {len(users)} users")
        return users
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")


@router.get("/users/{uid}", response_model=UserDetailResponse)
async def get_user_detail(uid: str) -> UserDetailResponse:
    """Get user detail with assigned cameras."""
    db = _get_db()
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        user_doc = db.collection('users').document(uid).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        data = user_doc.to_dict()
        
        # Check if user is disabled in Firebase Auth
        disabled = False
        try:
            firebase_user = auth.get_user(uid)
            disabled = firebase_user.disabled
        except Exception:
            pass
        
        cameras = _get_cameras_for_user(uid)
        
        return UserDetailResponse(
            uid=uid,
            email=data.get("email", ""),
            displayName=data.get("displayName"),
            authProvider=data.get("authProvider"),
            createdAt=str(data.get("createdAt")) if data.get("createdAt") else None,
            photoUrl=data.get("photoUrl"),
            disabled=disabled,
            camerasCount=len(cameras),
            cameras=cameras
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user")


@router.patch("/users/{uid}/status")
async def update_user_status(uid: str, request: UpdateUserStatusRequest) -> Dict[str, Any]:
    """Enable or disable a user in Firebase Auth."""
    try:
        auth.update_user(uid, disabled=request.disabled)
        
        status = "disabled" if request.disabled else "enabled"
        logger.info(f"User {uid} has been {status}")
        
        return {
            "message": f"User {status} successfully",
            "uid": uid,
            "disabled": request.disabled
        }
        
    except auth.UserNotFoundError:
        raise HTTPException(status_code=404, detail="User not found in Firebase Auth")
    except Exception as e:
        logger.error(f"Error updating user status: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user status")


# ============ Camera Endpoints ============

@router.get("/cameras", response_model=List[CameraResponse])
async def list_cameras() -> List[CameraResponse]:
    """Get all cameras (from hardcode config) with owner information from Firestore."""
    try:
        cameras = []
        
        # Use hardcoded camera list from config
        for cam_id in settings.rtsp_cameras:
            cam_info = CAMERA_LOCATIONS.get(cam_id, {"name": cam_id, "location": "Unknown"})
            owner_uid = _get_camera_owner(cam_id)
            owner_email = _get_user_email(owner_uid) if owner_uid else None
            
            cameras.append(CameraResponse(
                id=cam_id,
                name=cam_info["name"],
                location=cam_info["location"],
                owner_uid=owner_uid,
                owner_email=owner_email
            ))
        
        logger.info(f"Retrieved {len(cameras)} cameras")
        return cameras
        
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cameras")


@router.put("/cameras/{camera_id}/assign")
async def assign_camera(camera_id: str, request: AssignCameraRequest) -> Dict[str, Any]:
    """Assign or unassign a camera to a user."""
    db = _get_db()
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    # Validate camera exists in hardcode config
    if camera_id not in settings.rtsp_cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    try:
        # Validate user exists if assigning
        if request.user_uid:
            user_doc = db.collection('users').document(request.user_uid).get()
            if not user_doc.exists:
                raise HTTPException(status_code=404, detail="User not found")
        
        # Update or create camera document in Firestore
        camera_ref = db.collection('cameras').document(camera_id)
        camera_ref.set({
            "owner_uid": request.user_uid,
            "id": camera_id,
            "name": CAMERA_LOCATIONS.get(camera_id, {}).get("name", camera_id),
            "location": CAMERA_LOCATIONS.get(camera_id, {}).get("location", "Unknown"),
            "updatedAt": firestore.SERVER_TIMESTAMP
        }, merge=True)
        
        if request.user_uid:
            user_email = _get_user_email(request.user_uid)
            logger.info(f"Camera {camera_id} assigned to user {request.user_uid}")
            return {
                "message": "Camera assigned successfully",
                "camera_id": camera_id,
                "owner_uid": request.user_uid,
                "owner_email": user_email
            }
        else:
            logger.info(f"Camera {camera_id} unassigned")
            return {
                "message": "Camera unassigned successfully",
                "camera_id": camera_id,
                "owner_uid": None,
                "owner_email": None
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning camera: {e}")
        raise HTTPException(status_code=500, detail="Failed to assign camera")
