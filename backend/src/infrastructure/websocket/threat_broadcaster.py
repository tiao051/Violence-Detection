"""WebSocket threat detection broadcaster."""

import json
import logging
from typing import Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ThreatBroadcaster:
    """
    Manages WebSocket connections and broadcasts threat detection alerts.
    
    Maintains active WebSocket connections and sends real-time threat alerts
    to all connected clients when violence is detected.
    """
    
    def __init__(self):
        """Initialize threat broadcaster."""
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket) -> None:
        """
        Register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection instance
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection.
        
        Args:
            websocket: WebSocket connection instance to remove
        """
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast_threat(self, camera_id: str, detection: dict) -> None:
        """
        Broadcast threat detection to all connected clients.
        
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
        
        # Copy connections to avoid "Set changed size during iteration" error
        disconnected = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_status(self, threats: dict) -> None:
        """
        Broadcast current threat status for all cameras.
        
        Args:
            threats: Dictionary mapping camera_id to threat status
        """
        message = {
            "type": "threat_status",
            "threats": threats,
        }
        
        # Copy connections to avoid "Set changed size during iteration" error
        disconnected = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send status to client: {e}")
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)
    
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
