"""WebSocket threat detection broadcaster."""

import json
import logging
import os
from typing import Set, Dict, List, Optional
from fastapi import WebSocket
import jwt

logger = logging.getLogger(__name__)

# JWT configuration from environment
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY must be set in environment variables")


class AuthenticatedWebSocket:
    """Wrapper for WebSocket with authentication info."""
    
    def __init__(self, websocket: WebSocket, user_id: str, assigned_cameras: List[str]):
        """
        Initialize authenticated WebSocket.
        
        Args:
            websocket: FastAPI WebSocket instance
            user_id: Authenticated user ID
            assigned_cameras: List of camera IDs user has access to
        """
        self.websocket = websocket
        self.user_id = user_id
        self.assigned_cameras = assigned_cameras
    
    def has_access_to_camera(self, camera_id: str) -> bool:
        """Check if user has access to camera."""
        return camera_id in self.assigned_cameras
    
    async def send_json(self, data: dict) -> None:
        """Send JSON message to client."""
        await self.websocket.send_json(data)
    
    async def receive_json(self) -> dict:
        """Receive JSON message from client."""
        return await self.websocket.receive_json()
    
    async def close(self) -> None:
        """Close WebSocket connection."""
        await self.websocket.close()


class ThreatBroadcaster:
    """
    Manages WebSocket connections and broadcasts threat detection alerts.
    
    Maintains active WebSocket connections with authentication and broadcasts
    threat alerts only to clients who have access to the specific camera.
    """
    
    def __init__(self):
        """Initialize threat broadcaster."""
        self.active_connections: Set[AuthenticatedWebSocket] = set()
    
    def _extract_auth_token(self, query_params: Dict[str, List[str]]) -> Optional[str]:
        """
        Extract JWT token from query parameters or headers.
        
        Args:
            query_params: WebSocket query parameters
            
        Returns:
            JWT token if present, None otherwise
        """
        if "token" in query_params:
            return query_params["token"][0]
        return None
    
    def _verify_jwt_token(self, token: str) -> Optional[Dict]:
        """
        Verify JWT token and extract user info.
        
        Args:
            token: JWT token to verify
            
        Returns:
            User info dict with keys: uid, owner_cameras, if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Verify token type is 'access'
            if payload.get("type") != "access":
                return None
            
            return {
                "uid": payload.get("sub"),
                "owner_cameras": payload.get("owner_cameras", []),
            }
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def _get_user_assigned_cameras(self, user_id: str) -> List[str]:
        """
        Get list of cameras owned by user.
        
        Note: In the owner-only model, we get this from JWT token, not from DB.
        This method is kept for reference but shouldn't be called in normal flow.
        
        Args:
            user_id: Firebase user ID
            
        Returns:
            List of camera IDs user owns
        """
        # In owner-only model, this info comes from JWT
        # The JWT token already contains owner_cameras list
        return []  # Placeholder - actual cameras from JWT
    
    async def connect(self, websocket: WebSocket, token: Optional[str] = None) -> Optional[AuthenticatedWebSocket]:
        """
        Register a new WebSocket connection with JWT authentication.
        
        Args:
            websocket: WebSocket connection instance
            token: JWT authentication token
            
        Returns:
            AuthenticatedWebSocket if successful, None if authentication failed
        """
        # Verify JWT token
        if not token:
            logger.warning("WebSocket connection attempted without token")
            await websocket.close(code=1008, reason="Authentication required")
            return None
        
        user_info = self._verify_jwt_token(token)
        if not user_info:
            logger.warning("WebSocket connection attempted with invalid token")
            await websocket.close(code=1008, reason="Invalid authentication token")
            return None
        
        user_id = user_info.get("uid")
        owner_cameras = user_info.get("owner_cameras", [])
        
        if not user_id:
            logger.warning("Token missing user ID")
            await websocket.close(code=1008, reason="Invalid user info")
            return None
        
        if not owner_cameras:
            logger.warning(f"User {user_id} owns no cameras")
            await websocket.close(code=1008, reason="No owned cameras")
            return None
        
        # Accept connection
        await websocket.accept()
        
        # Create authenticated WebSocket wrapper
        auth_ws = AuthenticatedWebSocket(websocket, user_id, owner_cameras)
        self.active_connections.add(auth_ws)
        
        logger.info(
            f"WebSocket authenticated for user {user_id} with cameras: {owner_cameras}. "
            f"Total connections: {len(self.active_connections)}"
        )
        
        return auth_ws
    
    def disconnect(self, websocket: AuthenticatedWebSocket) -> None:
        """
        Remove a WebSocket connection.
        
        Args:
            websocket: AuthenticatedWebSocket instance to remove
        """
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected for user {websocket.user_id}. "
                    f"Total connections: {len(self.active_connections)}")
    
    async def broadcast_threat(self, camera_id: str, detection: dict) -> None:
        """
        Broadcast threat detection to clients with access to the camera.
        
        Only sends to clients who have the specified camera_id in their assigned_cameras list.
        
        Args:
            camera_id: Camera identifier where threat was detected
            detection: Detection result dict with keys:
                - violence: bool
                - confidence: float
                - timestamp: float
        """
        message = {
            "type": "threat_detection",
            "camera_id": camera_id,
            "violence": detection.get('violence', False),
            "confidence": detection.get('confidence', 0.0),
            "timestamp": detection.get('timestamp', 0),
        }
        
        logger.debug(f"Broadcasting threat from camera {camera_id} to authorized clients")
        
        # Copy connections to avoid "Set changed size during iteration" error
        disconnected = []
        for auth_ws in list(self.active_connections):
            # Only send to clients with access to this camera
            if not auth_ws.has_access_to_camera(camera_id):
                logger.debug(f"User {auth_ws.user_id} not authorized for camera {camera_id}")
                continue
            
            try:
                await auth_ws.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to user {auth_ws.user_id}: {e}")
                disconnected.append(auth_ws)
        
        # Clean up disconnected clients
        for auth_ws in disconnected:
            self.disconnect(auth_ws)
    
    async def broadcast_status(self, threats: dict) -> None:
        """
        Broadcast current threat status for all cameras (filtered by user permissions).
        
        Args:
            threats: Dictionary mapping camera_id to threat status
        """
        # Copy connections to avoid "Set changed size during iteration" error
        disconnected = []
        for auth_ws in list(self.active_connections):
            # Filter threats by user's assigned cameras
            user_threats = {
                cam_id: threat for cam_id, threat in threats.items()
                if auth_ws.has_access_to_camera(cam_id)
            }
            
            if not user_threats:
                continue
            
            message = {
                "type": "threat_status",
                "threats": user_threats,
            }
            
            try:
                await auth_ws.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send status to user {auth_ws.user_id}: {e}")
                disconnected.append(auth_ws)
        
        for auth_ws in disconnected:
            self.disconnect(auth_ws)
    
    def get_connection_count(self) -> int:
        """
        Get number of active WebSocket connections.
        
        Returns:
            Number of connected clients
        """
        return len(self.active_connections)


# Global broadcaster instance
_broadcaster: ThreatBroadcaster = None


def get_threat_broadcaster() -> ThreatBroadcaster:
    """
    Get or create the global threat broadcaster singleton.
    
    Returns:
        ThreatBroadcaster instance
    """
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = ThreatBroadcaster()
    return _broadcaster
