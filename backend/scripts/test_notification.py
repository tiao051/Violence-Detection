import asyncio
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.firebase.setup import initialize_firebase
from src.infrastructure.storage.token_repository import get_token_repository
from src.infrastructure.notifications.notification_service import get_notification_service

async def main():
    print("\n--- STARTING BACKEND NOTIFICATION TEST ---\n")

    # 1. Initialize Firebase
    print("1. Initializing Firebase...")
    try:
        initialize_firebase()
        print("   ✅ Firebase initialized.")
    except Exception as e:
        print(f"   ❌ Failed to initialize Firebase: {e}")
        return

    # 2. Setup Test Data
    user_id = "H2399Gybu8TeslP8zyEyP4uhn0l2" # The user's specific UID
    fake_token = "fake_fcm_token_for_testing_12345" # Dummy token
    
    # 3. Test Token Registration
    print(f"\n2. Registering fake token for user: {user_id}...")
    repo = get_token_repository()
    try:
        success = await repo.save_token(user_id, fake_token)
        if success:
            print("   ✅ Token saved to Firestore.")
        else:
            print("   ❌ Failed to save token.")
    except Exception as e:
        print(f"   ❌ Error saving token: {e}")

    # 4. Test Token Retrieval
    print("\n3. Retrieving tokens from Firestore...")
    try:
        tokens = await repo.get_tokens(user_id)
        print(f"   Found {len(tokens)} tokens: {tokens}")
        
        if fake_token in tokens:
            print("   ✅ SUCCESS: Our fake token was retrieved.")
        else:
            print("   ❌ ERROR: Fake token not found in the list.")
    except Exception as e:
        print(f"   ❌ Error retrieving tokens: {e}")
        
    # 5. Test Sending Notification
    print("\n4. Sending test notification...")
    service = get_notification_service()
    try:
        # Note: This will likely return 0 successes because the token is fake,
        # but it verifies the code path and Firebase connection.
        count = service.send_multicast(
            tokens=[fake_token],
            title="Backend Test",
            body="If you see this, the notification service is working!",
            data={"test_id": "123"}
        )
        
        print(f"   Attempted to send. Success count: {count}")
        print("   (Note: Success count 0 is expected for a fake token. The important part is no crash!)")
        print("   ✅ Notification Service executed successfully.")
        
    except Exception as e:
        print(f"   ❌ Error sending notification: {e}")

    print("\n--- TEST COMPLETED ---")

if __name__ == "__main__":
    asyncio.run(main())
