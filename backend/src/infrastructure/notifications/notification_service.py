"""Notification Service for sending FCM messages."""

import logging
from typing import List, Dict, Any, Optional
import firebase_admin
from firebase_admin import messaging

logger = logging.getLogger(__name__)


class NotificationService:
    """
    Service to send push notifications via Firebase Cloud Messaging (FCM).
    """

    def send_to_token(
        self, 
        token: str, 
        title: str, 
        body: str, 
        data: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Send a notification to a single device token.
        """
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                data=data or {},
                token=token,
            )
            
            response = messaging.send(message)
            logger.info(f"Successfully sent message: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to token {token}: {e}")
            return False

    def send_multicast(
        self, 
        tokens: List[str], 
        title: str, 
        body: str, 
        data: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Send a notification to multiple device tokens.
        Returns the number of successful messages.
        """
        if not tokens:
            return 0
        
        success_count = 0
        
        # Send to each token individually (workaround for batch endpoint 404)
        for token in tokens:
            try:
                message = messaging.Message(
                    notification=messaging.Notification(
                        title=title,
                        body=body,
                    ),
                    data=data or {},
                    token=token,
                )
                
                response = messaging.send(message)
                logger.info(f"Successfully sent message to token: {token[:20]}...")
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error sending message to token {token[:20]}...: {e}")
        
        logger.info(f"Successfully sent {success_count}/{len(tokens)} messages")
        return success_count


# Singleton instance
_notification_service: Optional[NotificationService] = None

def get_notification_service() -> NotificationService:
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service
