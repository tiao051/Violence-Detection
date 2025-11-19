"""WebSocket routes for real-time threat detection streaming."""

import asyncio
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from typing import Set, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Keep track of active WebSocket connections
active_connections: Set[WebSocket] = set()


class ConnectionManager:
    """Manage WebSocket connections and broadcast threat data."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        """Remove a disconnected WebSocket."""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.warning(f"Error broadcasting to client: {e}")
                disconnected.add(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.discard(connection)

    async def send_personal(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send data to a specific WebSocket connection."""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.warning(f"Error sending personal message: {e}")


manager = ConnectionManager()


@router.websocket("/ws/threats")
async def websocket_threat_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time threat detection streaming.

    Clients connect and receive threat detection updates in real-time.
    The endpoint continuously streams metadata from Redis to connected clients.
    """
    await manager.connect(websocket)
    frame_counter = 0

    try:
        # Get redis_producer from app state (set during startup)
        redis_producer = websocket.app.state.redis_producer
        
        if not redis_producer:
            logger.error("Redis producer not available")
            await manager.send_personal(
                websocket, 
                {"type": "error", "message": "Redis producer not initialized"}
            )
            return

        # Continuously stream data to client (push model, not pull)
        while True:
            try:
                # Get redis client from app state
                redis_client = websocket.app.state.redis_producer.redis_client if hasattr(websocket.app.state, 'redis_producer') else None
                
                if not redis_client:
                    logger.error("Redis client not available")
                    break

                # Stream latest frame data from all cameras
                for camera_id in ['cam1', 'cam2', 'cam3', 'cam4']:
                    stream_key = f"frames:{camera_id}"
                    
                    try:
                        # Get latest message from stream
                        messages = await redis_client.xrevrange(stream_key, count=1)
                        
                        if messages:
                            msg_id, msg_data = messages[0]
                            frame_counter += 1
                            
                            # Extract detection info
                            detection_str = msg_data.get(b'detection', b'{}').decode()
                            detection = json.loads(detection_str) if detection_str else {}
                            
                            metadata = {
                                "type": "detection",
                                "frame_seq": frame_counter,
                                "camera_id": camera_id,
                                "timestamp": msg_data.get(b'ts', b'').decode(),
                                "violence": detection.get('violence', False),
                                "confidence": detection.get('confidence', 0.0),
                            }
                            
                            await manager.send_personal(websocket, metadata)
                    
                    except Exception as e:
                        logger.debug(f"Error reading stream {stream_key}: {e}")
                        continue

                # Small delay to avoid busy-waiting
                await asyncio.sleep(0.5)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error streaming threat data: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)
        logger.info("Client disconnected")

