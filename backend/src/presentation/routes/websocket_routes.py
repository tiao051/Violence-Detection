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
    Subscribes to Redis pub/sub channel for immediate alerts.
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

        # Subscribe to threat alerts channel
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("threat_alerts")
        
        try:
            while True:
                # Wait for messages from pub/sub channel
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                
                if message and message["type"] == "message":
                    try:
                        alert_data = json.loads(message["data"])
                        # Send alert to WebSocket client
                        await manager.send_personal(websocket, alert_data)
                        logger.debug(f"Sent threat alert to client: {alert_data}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse alert message: {e}")
                
                # Check if WebSocket is still connected
                if websocket.client_state != 1:  # 1 = CONNECTED
                    break

        except Exception as e:
            logger.error(f"Error in pub/sub loop: {e}")
        finally:
            await pubsub.unsubscribe("threat_alerts")
            await pubsub.close()

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)
        logger.info("Client disconnected")
