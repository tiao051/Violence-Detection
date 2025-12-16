"""WebSocket routes for real-time threat alerts."""

import asyncio
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
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
        logger.debug(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.debug(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal(self, websocket: WebSocket, data: Dict[str, Any]):
        try:
            await websocket.send_json(data)
        except RuntimeError as e:
            # WebSocket is already closed or connection lost
            if "Unexpected ASGI message" in str(e) or "websocket.close" in str(e):
                pass
            else:
                logger.debug(f"Error sending personal message (client likely disconnected): {e}")
        except Exception as e:
            # Suppress common disconnect errors
            if "1000" in str(e) or "1001" in str(e):
                pass
            else:
                logger.warning(f"Error sending personal message: {e}")


manager = ConnectionManager()


@router.websocket("/ws/threats")
async def websocket_threat_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time threat alerts.
    Subscribes to Redis pub/sub channel for immediate alerts.
    """
    await manager.connect(websocket)
    try:
        # Wait briefly for app startup to initialize redis_client
        redis_client = getattr(websocket.app.state, "redis_client", None)
        wait_ms = 0
        while not redis_client and wait_ms < 5000:
            await asyncio.sleep(0.25)
            wait_ms += 250
            redis_client = getattr(websocket.app.state, "redis_client", None)

        if not redis_client:
            logger.error("Redis client not available")
            try:
                await manager.send_personal(websocket, {"type": "error", "message": "Redis client unavailable"})
            except Exception:
                pass
            return

        # Subscribe to all camera alerts
        pubsub = redis_client.pubsub()
        await pubsub.psubscribe("alerts:*")
        
        try:
            while True:
                # Check if client is still connected
                if websocket.client_state == WebSocketState.DISCONNECTED:
                    break

                # Wait for messages from pub/sub channel
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                except Exception as e:
                    logger.error(f"Error fetching message from pubsub: {e}")
                    message = None

                if message and message.get("type") == "pmessage":
                    try:
                        # message["data"] may be bytes; ensure string
                        data = message.get("data")
                        if isinstance(data, (bytes, bytearray)):
                            data = data.decode("utf-8")
                        alert_data = json.loads(data)
                        
                        # Inject 'type' field if missing, required by frontend
                        if "type" not in alert_data:
                            alert_data["type"] = "alert"

                        # Send alert to WebSocket client
                        await manager.send_personal(websocket, alert_data)
                        logger.debug(f"Sent threat alert to client: {alert_data}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse alert message: {e}")
                
                await asyncio.sleep(0.01)

        except WebSocketDisconnect:
            logger.debug("Client disconnected normally")
        except Exception as e:
            logger.error(f"Error in pub/sub loop: {e}")
        finally:
            await pubsub.punsubscribe("alerts:*")
            await pubsub.close()

    except WebSocketDisconnect:
        logger.debug("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)
