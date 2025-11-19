import 'package:flutter_dotenv/flutter_dotenv.dart';

/// Application configuration loaded from .env file.
/// 
/// The .env file should be loaded in main() before AppConfig is accessed:
///   await dotenv.load(fileName: ".env");
class AppConfig {
  // Backend API Configuration - loaded from .env
  static String get backendUrl => 
    dotenv.env['BACKEND_URL'] ?? 'http://localhost:8000';

  static String get backendWsUrl => 
    dotenv.env['BACKEND_WS_URL'] ?? 'ws://localhost:8000';

  // API Endpoints
  static String get authVerifyFirebaseUrl => '$backendUrl/api/v1/auth/verify-firebase';
  static String get authRefreshTokenUrl => '$backendUrl/api/v1/auth/refresh';
  static String get camerasListUrl => '$backendUrl/api/v1/auth/cameras';
  static String cameraStreamUrl(String cameraId) => '$backendUrl/api/v1/auth/streams/$cameraId/url';
}
