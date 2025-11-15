import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';

/// Callback type for handling notification tap events.
/// 
/// Called when user taps a notification while app is in foreground or background.
/// The [eventId] parameter is extracted from the notification data.
typedef OnNotificationTapCallback = void Function(String eventId);

/// Simple helper for Firebase Cloud Messaging (FCM) setup used in development.
///
/// Responsibilities:
///  - Request runtime notification permissions (iOS & Android 13+).
///  - Retrieve the device FCM token so the backend can target this device.
///  - Setup handlers for foreground and background notification taps.
///
/// Notes:
///  - This class intentionally keeps logic minimal and side-effect free beyond
///    requesting permissions and printing the token in debug mode. Sending the
///    token to a backend or persisting it should be implemented by the
///    application (see the TODO below).
class NotificationService {
  final FirebaseMessaging _firebaseMessaging = FirebaseMessaging.instance;
  static OnNotificationTapCallback? _onNotificationTap;

  /// Initialize notification capabilities with optional deep link handler.
  ///
  /// [onNotificationTap] is called when user taps a notification. 
  /// This enables routing to event details via deep linking.
  ///
  /// This requests permissions when appropriate and retrieves the device's
  /// FCM token. Call this early (e.g., app startup) but after necessary
  /// platform bindings are ready.
  Future<void> initialize({OnNotificationTapCallback? onNotificationTap}) async {
    _onNotificationTap = onNotificationTap;
    await _requestPermission();
    await _getToken();
    await _setupMessageHandlers();
  }

  /// Setup handlers for foreground and background notifications.
  ///
  /// Foreground: User sees notification while app is open.
  /// Background: User taps notification from system tray while app is closed or backgrounded.
  Future<void> _setupMessageHandlers() async {
    // Handle foreground messages
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      if (kDebugMode) {
        print('NotificationService: Foreground message received');
        print('Title: ${message.notification?.title}');
        print('Body: ${message.notification?.body}');
        print('Data: ${message.data}');
      }
      // In foreground, system notification won't show automatically on Android.
      // You could display a custom notification or handle it here.
    });

    // Handle notification tap when app is in foreground
    FirebaseMessaging.onMessageOpenedApp.listen((RemoteMessage message) {
      if (kDebugMode) {
        print('NotificationService: Notification tapped (foreground)');
      }
      _handleNotificationTap(message);
    });

    // Handle initial message when app is opened from terminated state
    final initialMessage = await _firebaseMessaging.getInitialMessage();
    if (initialMessage != null) {
      if (kDebugMode) {
        print('NotificationService: App opened from terminated state via notification');
      }
      _handleNotificationTap(initialMessage);
    }
  }

  /// Handle notification tap by extracting event ID and calling callback.
  static void _handleNotificationTap(RemoteMessage message) {
    // Extract event_id from notification data
    final eventId = message.data['event_id'];
    
    if (eventId != null && _onNotificationTap != null) {
      if (kDebugMode) {
        print('NotificationService: Deep linking to event: $eventId');
      }
      _onNotificationTap!(eventId);
    } else {
      if (kDebugMode) {
        print('NotificationService: No event_id in notification data or callback not set');
      }
    }
  }

  /// Request notification permission from the user.
  ///
  /// We ask for alert/badge/sound permission on platforms that require
  /// explicit consent (iOS / Android 13+). We log the resulting status in
  /// debug mode to help during development.
  Future<void> _requestPermission() async {
    try {
      final NotificationSettings settings =
          await _firebaseMessaging.requestPermission(
        alert: true,
        badge: true,
        sound: true,
        announcement: false,
        carPlay: false,
        criticalAlert: false,
        provisional: false,
      );

      if (kDebugMode) {
        switch (settings.authorizationStatus) {
          case AuthorizationStatus.authorized:
            print('NotificationService: Permission granted.');
            break;
          case AuthorizationStatus.provisional:
            print('NotificationService: Provisional permission granted.');
            break;
          case AuthorizationStatus.denied:
          case AuthorizationStatus.notDetermined:
            print('NotificationService: Permission declined or not determined.');
            break;
        }
      }
    } catch (e) {
      // Non-fatal: log error and continue. Permission failure should not crash app.
      if (kDebugMode) {
        print('NotificationService: Failed to request permission: $e');
      }
    }
  }

  /// Retrieve the FCM token for this device.
  ///
  /// The token uniquely identifies the app+device instance. In production
  /// you should send this token to your backend so it can send targeted
  /// notifications. We only print the token in debug builds to avoid leaking
  /// sensitive identifiers in release logs.
  Future<void> _getToken() async {
    try {
      final token = await _firebaseMessaging.getToken();

      if (token == null) {
        if (kDebugMode) {
          print('NotificationService: Unable to retrieve FCM token (null).');
        }
        return;
      }

      if (kDebugMode) {
        // Clear, human-friendly log block for easier discovery during dev
        print('--- FCM TOKEN START ----------------------------------');
        print('FCM TOKEN: $token');
        print('--- FCM TOKEN END ------------------------------------');
      }

      // TODO: Send this token to your backend and associate it with the
      // authenticated user/device. Example:
      //   await backendApi.registerDeviceToken(token);
      // Keep this class focused on FCM interactions; let the app decide how
      // to persist or transmit the token.
    } catch (e) {
      if (kDebugMode) {
        print('NotificationService: Failed to get FCM token: $e');
      }
    }
  }
}