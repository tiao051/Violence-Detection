"""WebSocket routes for real-time threat alerts."""

import asyncio
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manage WebSocket connections and broadcast threat alerts."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal(self, websocket: WebSocket, data: Dict[str, Any]):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.warning(f"Error sending personal message: {e}")

    async def broadcast(self, data: Dict[str, Any]):
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.warning(f"Error broadcasting to client: {e}")
                disconnected.add(connection)
        for connection in disconnected:
            self.active_connections.discard(connection)


manager = ConnectionManager()


@router.websocket("/ws/threats")
async def websocket_threat_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time threat alerts.
    Only sends metadata when violence is detected.
    """
    await manager.connect(websocket)
    try:
        redis_producer = websocket.app.state.redis_producer
        if not redis_producer:
            logger.error("Redis producer not available")
            await manager.send_personal(
                websocket, {"type": "error", "message": "Redis producer not initialized"}
            )
            return

        redis_client = getattr(redis_producer, 'redis_client', None)
        if not redis_client:
            logger.error("Redis client not available")
            return

        while True:
            try:
                # Check each camera
                for camera_id in ['cam1', 'cam2', 'cam3', 'cam4']:
                    stream_key = f"frames:{camera_id}"
                    messages = await redis_client.xrevrange(stream_key, count=1)

                    if messages:
                        msg_id, msg_data = messages[0]

                        detection_str = msg_data.get(b'detection', b'{}').decode()
                        detection = json.loads(detection_str) if detection_str else {}

                        # Only send if violence detected
                        if detection.get("violence", False):
                            metadata = {
                                "type": "alert",
                                "camera_id": camera_id,
                                "timestamp": msg_data.get(b'ts', b'').decode(),
                                "confidence": detection.get("confidence", 0.0),
                            }
                            await manager.send_personal(websocket, metadata)

                await asyncio.sleep(0.5)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error streaming threat alerts: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)
        logger.info("Client disconnected")
