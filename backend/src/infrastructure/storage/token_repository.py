"""Repository for managing FCM tokens."""

import logging
from typing import List, Optional
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

logger = logging.getLogger(__name__)


class TokenRepository:
    """
    Repository for storing and retrieving FCM tokens in Firestore.
    Structure: users/{user_id}/fcm_tokens/{token_id}
    """

    def __init__(self):
        try:
            self.db = firestore.client()
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            self.db = None

    async def save_token(self, user_id: str, token: str, device_type: str = "android") -> bool:
        """
        Save or update an FCM token for a user.
        """
        if not self.db:
            return False

        try:
            # Reference to the user's token collection
            tokens_ref = self.db.collection('users').document(user_id).collection('fcm_tokens')
            
            # Check if token already exists to avoid duplicates
            # We use the token string itself as the document ID to ensure uniqueness easily
            # or we can query. Using token as ID is simpler for deduplication.
            # However, token strings can be long. Let's query first.
            
            # Strategy: Use the token as the document ID. 
            # Firebase document IDs have a limit, but FCM tokens are usually within limits (though long).
            # Safer approach: Query for the token.
            
            query = tokens_ref.where(filter=FieldFilter("token", "==", token)).limit(1)
            docs = query.stream()
            
            existing_doc = None
            for doc in docs:
                existing_doc = doc
                break
            
            data = {
                "token": token,
                "deviceType": device_type,
                "updatedAt": firestore.SERVER_TIMESTAMP,
                "isActive": True
            }
            
            if existing_doc:
                # Update existing
                existing_doc.reference.update(data)
                logger.info(f"Updated existing FCM token for user {user_id}")
            else:
                # Create new
                tokens_ref.add(data)
                logger.info(f"Saved new FCM token for user {user_id}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving FCM token: {e}")
            return False

    async def get_tokens(self, user_id: str) -> List[str]:
        """
        Get all active FCM tokens for a user.
        """
        if not self.db:
            return []

        try:
            tokens_ref = self.db.collection('users').document(user_id).collection('fcm_tokens')
            query = tokens_ref.where(filter=FieldFilter("isActive", "==", True))
            docs = query.stream()
            
            tokens = []
            for doc in docs:
                data = doc.to_dict()
                token = data.get("token")
                if token:
                    tokens.append(token)
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error retrieving tokens for user {user_id}: {e}")
            return []


# Singleton instance
_token_repository: Optional[TokenRepository] = None

def get_token_repository() -> TokenRepository:
    global _token_repository
    if _token_repository is None:
        _token_repository = TokenRepository()
    return _token_repository
