# API Specification for Backend Developer

## Overview
This document specifies all API endpoints required for the **Violence Detection Security App** (Flutter mobile client). 

**Base URL:** `https://your-api-domain.com/api/v1` (please provide)

**Authentication:** All endpoints (except `/login`) require Bearer token authentication.
- Header: `Authorization: Bearer <token>`

**Response Format:** All responses must be in JSON format with proper HTTP status codes.

---

## 1. Authentication

### 1.1 Login
**Endpoint:** `POST /login`

**Description:** Authenticate user and return access token

**Authentication Required:** No

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Success Response (200 OK):**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "user_001",
    "name": "Admin User",
    "email": "admin@example.com"
  }
}
```

**Error Response (401 Unauthorized):**
```json
{
  "error": "Invalid credentials"
}
```

---

### 1.2 Google Sign-In
**Endpoint:** `POST /auth/google`

**Description:** Authenticate user with Google OAuth token and return access token

**Authentication Required:** No

**Request Body:**
```json
{
  "googleToken": "eyJhbGciOiJSUzI1NiIsImtpZCI6IjEifQ..."
}
```

**Fields:**
- `googleToken` (string, required): Google OAuth ID token from Google Sign-In SDK

**Success Response (200 OK):**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "user_002",
    "name": "John Doe",
    "email": "john@gmail.com",
    "isNewUser": false
  }
}
```

**Success Response - First Time User (201 Created):**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "user_999",
    "name": "Jane Smith",
    "email": "jane@gmail.com",
    "isNewUser": true
  }
}
```

**Error Response (401 Unauthorized):**
```json
{
  "error": "Invalid or expired Google token"
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "Google token verification failed",
  "details": "Token signature invalid or expired"
}
```

**Notes for Backend:**
- Verify Google token using Google's public key (from Google's JWKS endpoint)
- Check token is not expired
- Extract user information (email, name) from token claims
- If user with email doesn't exist in database, create new user automatically
- Return `isNewUser: true` for newly created accounts (HTTP 201)
- Return `isNewUser: false` for existing accounts (HTTP 200)
- Use same JWT token format as username/password login
- **Important:** Any valid Google account can be used to sign in - no whitelist needed
  - Users can use @gmail.com accounts
  - Users can use @company.com (Google Workspace) accounts
  - Users can use any email domain that has Google account
- Store email as unique identifier (to recognize returning users)
- Extract and store: email, displayName (from Google token)

**Google Token Verification Flow:**
1. Receive `googleToken` from mobile app
2. Call Google's tokeninfo endpoint OR verify locally using Google's public keys:
   - Endpoint: `https://www.googleapis.com/oauth2/v1/tokeninfo?id_token={token}`
   - OR locally: Verify JWT signature using key from `https://www.googleapis.com/oauth2/v1/certs`
3. Check `aud` (audience) claim matches your Google OAuth 2.0 client ID
4. Check `exp` (expiration) is in the future
5. Extract: `email`, `name`, `email_verified`
6. Look up user by email in database:
   - If found: Return existing user token + `isNewUser: false` (200 OK)
   - If not found: Create new user + Return token + `isNewUser: true` (201 Created)

**Example Google Token Claims (decoded):**
```json
{
  "iss": "https://accounts.google.com",
  "azp": "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com",
  "aud": "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com",
  "sub": "107693717500...",
  "email": "user@gmail.com",
  "email_verified": true,
  "name": "John Doe",
  "picture": "https://lh3.googleusercontent.com/...",
  "given_name": "John",
  "family_name": "Doe",
  "iat": 1700000000,
  "exp": 1700003600
}
```

---

### 1.3 Firebase Sign-In Verification
**Endpoint:** `POST /auth/firebase-verify`

**Description:** Verify Firebase ID token and return app JWT token

**Authentication Required:** No

**Request Body:**
```json
{
  "firebaseToken": "eyJhbGciOiJSUzI1NiIsImtpZCI6IjEifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20iLCJhdWQiOiJzZWN1cml0eS1hcHAtYmU5MjciLCJhdXRoX3RpbWUiOjE3MzE2NDAwMjUsInVzZXJfaWQiOiI0WWdTRTRxZWFQZ05jNWw5UDlpa3V2WmZmQlkyIiwic3ViIjoiNFlnU0U0cWVhUGdOYzVsOVA5aWt1dlpmZkJZMiIsImlhdCI6MTczMTY0MDAxNSwiZXhwIjoxNzMxNjQzNjE1LCJlbWFpbCI6Im5hbW5ndXllbmRlcHRyYWk1NjhAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnsiZ29vZ2xlLmNvbSI6WyIxMDc2OTM3MTc1MDAzMjEwODI5NzEiXSwiZW1haWwiOlsibmFtbmd1eWVuZGVwdHJhaTU2OEBnbWFpbC5jb20iXX0sInNpZ25faW5fcHJvdmlkZXIiOiJnb29nbGUuY29tIn19.signature..."
}
```

**Fields:**
- `firebaseToken` (string, required): Firebase ID token from mobile app (JWT signed by Firebase)

**Success Response (200 OK - New User):**
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0WWdTRTRxZWFQZ05jNWw5UDlpa3V2WmZmQlkyIiwiaWF0IjoxNzMxNjQwMDAwLCJleHAiOjE3MzE3MjY0MDB9.signature",
  "user": {
    "id": "user_firebase_4YgSE4qeaPgNc5l9P9ikuVZfvBY2",
    "email": "namnguyendeptrai568@gmail.com",
    "isNewUser": true
  }
}
```

**Success Response (200 OK - Existing User):**
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0WWdTRTRxZWFQZ05jNWw5UDlpa3V2WmZmQlkyIiwiaWF0IjoxNzMxNjQwMDAwLCJleHAiOjE3MzE3MjY0MDB9.signature",
  "user": {
    "id": "user_firebase_4YgSE4qeaPgNc5l9P9ikuVZfvBY2",
    "email": "namnguyendeptrai568@gmail.com",
    "isNewUser": false
  }
}
```

**Error Response (401 Unauthorized):**
```json
{
  "error": "Invalid Firebase token",
  "details": "Firebase token verification failed or expired"
}
```

**Implementation Notes for Backend:**
- Use Firebase Admin SDK to verify Firebase ID token (available for Node.js, Python, Java, Go, C#, etc):
  ```python
  # Python example using firebase-admin
  import firebase_admin
  from firebase_admin import auth
  
  def verify_firebase_token(firebase_token):
    try:
      decoded_token = auth.verify_id_token(firebase_token)
      firebase_uid = decoded_token['uid']
      email = decoded_token['email']
      
      # Check if user exists in database using Firebase UID
      user = db.users.find_one({'firebaseUid': firebase_uid})
      
      if user:
        is_new_user = False
      else:
        # Create new user
        user = {
          'firebaseUid': firebase_uid,
          'email': email,
          'createdAt': datetime.now()
        }
        db.users.insert_one(user)
        is_new_user = True
      
      # Generate app JWT token (with your app's secret key)
      app_jwt = generate_jwt({
        'sub': firebase_uid,
        'email': email,
        'userId': str(user['_id'])
      })
      
      return {
        'success': True,
        'token': app_jwt,
        'user': {
          'id': f"user_firebase_{firebase_uid}",
          'email': email,
          'isNewUser': is_new_user
        }
      }
    except auth.ExpiredSignError:
      return 401, {'error': 'Token expired'}
    except auth.InvalidIdTokenError:
      return 401, {'error': 'Invalid token'}
  ```

- Firebase token is signed with Firebase private key, backend verifies signature using Firebase public keys
- Check token is not expired (firebase_admin SDK does this automatically)
- Extract user UID from decoded token (available as `uid` or `sub`)
- Extract email from decoded token
- Use Firebase UID as unique user identifier (stable across app lifetime)
- Create user in database if first time login (based on Firebase UID)
- Generate app JWT token with:
  - `sub` claim: Firebase UID
  - `email` claim: User email
  - `exp` claim: Token expiration time (recommend 7 days)
- Return app JWT for subsequent API calls (instead of Firebase token)
- All endpoints except login require: `Authorization: Bearer <app_jwt>`

---

## 2. Cameras

### 2.1 Get All Cameras
**Endpoint:** `GET /cameras`

**Description:** Retrieve list of all security cameras

**Authentication Required:** Yes

**Query Parameters:** None

**Success Response (200 OK):**
```json
[
  {
    "id": "cam_001",
    "name": "Front Gate Camera",
    "location": "Entrance",
    "status": "online",
    "streamUrl": "rtsp://camera-server.com/stream1",
    "thumbnailUrl": "https://cdn.example.com/thumbnails/cam_001.jpg"
  },
  {
    "id": "cam_002",
    "name": "Parking Lot Camera",
    "location": "Parking Area",
    "status": "offline",
    "streamUrl": "rtsp://camera-server.com/stream2",
    "thumbnailUrl": "https://cdn.example.com/thumbnails/cam_002.jpg"
  }
]
```

**Required Fields:**
- `id` (string): Unique camera identifier
- `name` (string): Display name
- `status` (string): "online" or "offline"
- `streamUrl` (string): Camera stream URL (RTSP, HLS, or HTTP)

**Optional Fields:**
- `location` (string): Physical location description
- `thumbnailUrl` (string): Thumbnail image URL

---

### 2.2 Get Camera Stream URL
**Endpoint:** `GET /cameras/{id}/stream`

**Description:** Get streaming URL for specific camera (if different from list endpoint)

**Authentication Required:** Yes

**Path Parameters:**
- `id` (string): Camera ID

**Success Response (200 OK):**
```json
{
  "streamUrl": "rtsp://camera-server.com/stream1",
  "streamType": "rtsp"
}
```

**Error Response (404 Not Found):**
```json
{
  "error": "Camera not found"
}
```

---

## 3. Events (Detected Violations)

### 3.1 Get All Events
**Endpoint:** `GET /events`

**Description:** Retrieve list of all detected violation events

**Authentication Required:** Yes

**Query Parameters (optional):**
- `limit` (integer): Number of events to return (default: 50)
- `offset` (integer): Pagination offset (default: 0)
- `startDate` (ISO 8601): Filter events from this date
- `endDate` (ISO 8601): Filter events until this date
- `cameraId` (string): Filter by specific camera

**Success Response (200 OK):**
```json
[
  {
    "id": "evt_001",
    "cameraId": "cam_001",
    "cameraName": "Front Gate Camera",
    "timestamp": "2025-11-14T07:56:00Z",
    "isViewed": false,
    "thumbnailUrl": "https://cdn.example.com/events/evt_001_thumb.jpg",
    "videoUrl": "https://cdn.example.com/events/evt_001_video.mp4",
    "severity": "high",
    "detectionType": "violence"
  },
  {
    "id": "evt_002",
    "cameraId": "cam_002",
    "cameraName": "Parking Lot Camera",
    "timestamp": "2025-11-14T06:30:00Z",
    "isViewed": true,
    "thumbnailUrl": "https://cdn.example.com/events/evt_002_thumb.jpg",
    "videoUrl": "https://cdn.example.com/events/evt_002_video.mp4",
    "severity": "medium",
    "detectionType": "violence"
  }
]
```

**Required Fields:**
- `id` (string): Unique event identifier
- `cameraId` (string): Camera that detected the event
- `cameraName` (string): Camera display name
- `timestamp` (ISO 8601): When event occurred
- `isViewed` (boolean): Whether user has viewed this event
- `videoUrl` (string): URL to event video clip

**Optional Fields:**
- `thumbnailUrl` (string): Event thumbnail image
- `severity` (string): "low", "medium", "high"
- `detectionType` (string): Type of detection

---

### 3.2 Get Event Detail
**Endpoint:** `GET /events/{id}`

**Description:** Get detailed information about specific event

**Authentication Required:** Yes

**Path Parameters:**
- `id` (string): Event ID

**Success Response (200 OK):**
```json
{
  "id": "evt_001",
  "cameraId": "cam_001",
  "cameraName": "Front Gate Camera",
  "timestamp": "2025-11-14T07:56:00Z",
  "isViewed": false,
  "thumbnailUrl": "https://cdn.example.com/events/evt_001_thumb.jpg",
  "videoUrl": "https://cdn.example.com/events/evt_001_video.mp4",
  "duration": 15,
  "severity": "high",
  "detectionType": "violence",
  "confidence": 0.95
}
```

**Error Response (404 Not Found):**
```json
{
  "error": "Event not found"
}
```

---

### 3.3 Mark Event as Viewed
**Endpoint:** `PATCH /events/{id}/viewed`

**Description:** Mark event as viewed by user (used for badge count)

**Authentication Required:** Yes

**Path Parameters:**
- `id` (string): Event ID

**Request Body:** Empty or
```json
{
  "isViewed": true
}
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "message": "Event marked as viewed"
}
```

**Error Response (404 Not Found):**
```json
{
  "error": "Event not found"
}
```

---

### 3.4 Report False Detection
**Endpoint:** `POST /events/{id}/report`

**Description:** User reports this event as false detection (to improve ML model)

**Authentication Required:** Yes

**Path Parameters:**
- `id` (string): Event ID

**Request Body:**
```json
{
  "reason": "No violence detected, normal activity"
}
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "message": "Report submitted successfully"
}
```

**Error Response (404 Not Found):**
```json
{
  "error": "Event not found"
}
```

---

## 4. User Settings

### 4.1 Get User Settings
**Endpoint:** `GET /settings`

**Description:** Retrieve current user's app settings/preferences

**Authentication Required:** Yes

**Success Response (200 OK):**
```json
{
  "enableMotionAlerts": true,
  "alertSensitivity": 70,
  "alertSound": "default",
  "darkMode": false,
  "refreshRateSeconds": 30,
  "enableNotifications": true,
  "showBadge": true
}
```

**All Fields:**
- `enableMotionAlerts` (boolean): Enable/disable motion detection alerts
- `alertSensitivity` (integer, 0-100): Detection sensitivity level
- `alertSound` (string): Sound to play on alert ("default", "chime", "siren", "silent")
- `darkMode` (boolean): Enable dark mode UI
- `refreshRateSeconds` (integer): How often to refresh data (15, 30, 60 seconds)
- `enableNotifications` (boolean): Enable push notifications
- `showBadge` (boolean): Show badge count for unviewed events

**Note:** All fields are required. If user has no settings saved, return default values shown above.

---

### 4.2 Update User Settings
**Endpoint:** `POST /settings`

**Description:** Update user's app settings (can update all fields or partial)

**Authentication Required:** Yes

**Request Body (example - partial update allowed):**
```json
{
  "alertSensitivity": 80,
  "darkMode": true,
  "refreshRateSeconds": 60
}
```

**Request Body (example - full update):**
```json
{
  "enableMotionAlerts": true,
  "alertSensitivity": 80,
  "alertSound": "chime",
  "darkMode": true,
  "refreshRateSeconds": 60,
  "enableNotifications": true,
  "showBadge": false
}
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "message": "Settings updated successfully",
  "settings": {
    "enableMotionAlerts": true,
    "alertSensitivity": 80,
    "alertSound": "chime",
    "darkMode": true,
    "refreshRateSeconds": 60,
    "enableNotifications": true,
    "showBadge": false
  }
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "Invalid field value",
  "details": "alertSensitivity must be between 0 and 100"
}
```

---

## 5. Push Notifications (FCM)

### 5.1 Register FCM Token
**Endpoint:** `POST /users/{userId}/fcm-token`

**Description:** Register device's Firebase Cloud Messaging token for push notifications

**Authentication Required:** Yes

**Path Parameters:**
- `userId` (string): User ID (from login response)

**Request Body:**
```json
{
  "fcmToken": "eXaMpLe_FcM_ToKeN_123456789abcdef...",
  "deviceType": "android"
}
```

**Fields:**
- `fcmToken` (string, required): Firebase Cloud Messaging device token
- `deviceType` (string, optional): Device platform ("android", "ios", "web")

**Success Response (200 OK):**
```json
{
  "success": true,
  "message": "FCM token registered successfully"
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "Invalid FCM token format"
}
```

**Notes for Backend:**
- Use this token to send push notifications when new events are detected
- Token should be associated with the authenticated user
- Old tokens for same user+device should be replaced
- Notification payload should include event details:
  ```json
  {
    "notification": {
      "title": "Violence Detected",
      "body": "Front Gate Camera - Click to view"
    },
    "data": {
      "eventId": "evt_001",
      "cameraId": "cam_001",
      "type": "violence_detection"
    }
  }
  ```

---

## 6. Authentication Management

### 6.1 Logout
**Endpoint:** `POST /logout`

**Description:** Logout user and invalidate current JWT token (optional endpoint)

**Authentication Required:** Yes

**Request Body:** Empty or
```json
{
  "fcmToken": "eXaMpLe_FcM_ToKeN_..."
}
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

**Notes:**
- If JWT is stateless, this endpoint is optional
- If implemented, add token to blacklist or remove from active sessions
- If `fcmToken` provided, unregister it from push notifications

---

## 7. Common HTTP Status Codes

Use these standard HTTP status codes for all API responses:

| Status Code | Meaning | Usage |
|-------------|---------|-------|
| **200 OK** | Success | Request completed successfully |
| **201 Created** | Resource Created | New resource created (optional - can use 200) |
| **400 Bad Request** | Invalid Input | Missing required fields, invalid data format |
| **401 Unauthorized** | Authentication Failed | Invalid/missing token, wrong credentials |
| **403 Forbidden** | Access Denied | Valid token but insufficient permissions |
| **404 Not Found** | Resource Not Found | Camera/event/user doesn't exist |
| **500 Internal Server Error** | Server Error | Unexpected server-side error |

---

## 8. Error Response Format

All error responses should follow this consistent format:

```json
{
  "error": "Brief error message",
  "details": "More detailed explanation (optional)",
  "code": "ERROR_CODE_NAME (optional)"
}
```

**Examples:**

```json
{
  "error": "Invalid credentials",
  "details": "Username or password is incorrect"
}
```

```json
{
  "error": "Camera not found",
  "details": "No camera exists with ID: cam_999"
}
```

```json
{
  "error": "Validation failed",
  "details": "Field 'alertSensitivity' must be between 0 and 100"
}
```

---

## 9. Implementation Checklist for Backend Developer

Please ensure the following when implementing these APIs:

### Authentication
- [ ] JWT tokens generated on successful login
- [ ] Token expiration time (recommended: 7 days)
- [ ] Token validation on all protected endpoints
- [ ] Return user ID in login response (needed for FCM registration)

### Cameras
- [ ] Return both required fields (id, name, status, streamUrl)
- [ ] Optional fields (location, thumbnailUrl) can be added later
- [ ] Support for online/offline status
- [ ] Valid RTSP/HTTP stream URLs

### Events
- [ ] Include `isViewed` field (default: false for new events)
- [ ] Support date range filtering with query parameters
- [ ] Support camera filtering
- [ ] Video URLs must be accessible to mobile app
- [ ] Implement mark as viewed endpoint (for badge count)

### Settings
- [ ] Return default values if user has no saved settings
- [ ] Support partial updates (only changed fields)
- [ ] Validate field ranges (alertSensitivity: 0-100, refreshRateSeconds: 15/30/60)
- [ ] Store settings per user

### Push Notifications
- [ ] Store FCM tokens per user
- [ ] Send notification when new event detected
- [ ] Include event details in notification payload
- [ ] Handle token refresh/updates

### Testing
- [ ] Provide Postman collection or Swagger documentation
- [ ] Include example requests/responses
- [ ] Provide test credentials for development
- [ ] Test all error scenarios

---

## 10. Development Notes

**Base URL Format:**
```
https://your-api-domain.com/api/v1
```

**CORS Configuration:**
- Allow origins from mobile app domains
- Allow methods: GET, POST, PATCH, DELETE
- Allow headers: Authorization, Content-Type

**Security Recommendations:**
- Use HTTPS for all endpoints
- Implement rate limiting to prevent abuse
- Validate all input data
- Sanitize user inputs to prevent injection attacks
- Use environment variables for sensitive configuration

---

## 11. Questions & Contact

If you have any questions about these specifications:

1. **Unclear requirements?** → Re-read the endpoint description and examples
2. **Missing information?** → Use reasonable defaults, document your decision
3. **Technical questions?** → Contact the Flutter developer
4. **Suggestions for improvements?** → Propose them before implementing

**Flutter Developer Contact:** [Your contact info here]

---

**Thank you for implementing these APIs! Please notify when endpoints are ready for testing.**
