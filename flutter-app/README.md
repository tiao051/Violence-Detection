# Security Alert App (Flutter Frontend)

This is the official Flutter mobile application for the real-time Violence Detection project. This frontend provides a user interface for monitoring camera feeds, receiving critical alerts, and reviewing past security events.

*This project is currently a Work in Progress.*

---

## ✨ Core Features (MVP)

This version establishes the application's foundational structure. All API calls are currently mocked (simulated) via service classes in `lib/services/`.

### Authentication Flow
- **Login Screen:** Implemented with input validation.
- **Session Persistence:** Login tokens are saved to `SharedPreferences` to keep users logged in across sessions.
- **Logout Functionality:** Securely clears user session data.

### State Management
- **Centralized State:** Utilizes the `provider` package for clean and maintainable state management.
- **Dedicated Providers:** Separate providers created for `Auth`, `Camera`, and `Event` states to manage different parts of the app independently.

### Routing
- **Professional Routing:** Implemented using `go_router` for a robust, URL-based navigation system.
- **Route Guarding:** Protects sensitive routes from unauthenticated access by automatically redirecting users to the login screen.
- **Tab-based Navigation:** A `BottomNavigationBar` on the home screen allows easy switching between the main app sections.

### Camera Tab (Live View)
- Fetches a (mocked) list of available cameras from `CameraProvider`.
- Users can select a camera to open a dedicated `LiveViewScreen`.
- Streams (mocked) video feeds using the `video_player` package.
  - *(Note: Successfully replaced `flutter_vlc_player` after debugging native library issues, opting for a more stable solution).*

### Events Tab (Review & Feedback)
- Fetches a (mocked) list of past security incidents from `EventProvider`.
- Displays events in a clear list format with camera name and timestamp (formatted using `intl`).
- **Event Detail Screen:**
    - Allows users to tap an event to review the recorded (mocked) video clip.
    - **"Report False Positive" feature:**
        - A button with its own loading state (`isReporting`) provides clear user feedback.
        - Calls a (mocked) `POST /events/{id}/report` API endpoint.
        - Displays success or error messages using a `SnackBar`.

### Native Configuration
- **Android Setup:** Correctly configured `AndroidManifest.xml` with `INTERNET` permission and network security settings (`usesCleartextTraffic`) to enable video streaming from local development servers.

---

## 🚀 Tech Stack

- **Framework:** Flutter (Dart)
- **State Management:** `provider`
- **Routing:** `go_router`
- **Video Playback:** `video_player`
- **HTTP Requests:** `http`
- **Local Storage:** `shared_preferences`
- **Utilities:** `intl` (for date/time formatting)

---

## 🏁 Getting Started

### Prerequisites
- Flutter SDK (version 3.x or higher) installed.
- An Android emulator or a physical Android/iOS device.

### How to Run

1.  **Navigate to the app directory:**
    ```bash
    cd flutter-app
    ```

2.  **Clean the project (optional but recommended):**
    ```bash
    flutter clean
    ```

3.  **Install dependencies:**
    ```bash
    flutter pub get
    ```

4.  **Run the application:**
    ```bash
    flutter run
    ```

---

## 📂 Directory Structure

The project follows a feature-first architecture with a clear separation of concerns:

```
lib/
├── models/         # Data model classes (Camera, Event, User).
├── providers/      # State management logic using Provider.
├── screens/        # UI for individual screens and tabs.
├── services/       # Business logic and API communication (currently mocked).
├── utils/          # Helper functions and constants.
└── main.dart       # Application entry point and route configuration.
```

---

## 🔮 Next Steps

- **Backend Integration:** Replace all mocked service calls with live HTTP requests to the backend API.
- **Push Notifications:** Integrate Firebase Cloud Messaging (FCM) to handle real-time event alerts.
- **UI/UX Polishing:** Enhance the user interface with loading indicators, error dialogs, and a more refined design.
- **WebRTC Integration (Optional):** Explore using WebRTC for lower-latency live video streaming if required.