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

  // MediaMTX WHEP Streaming Configuration - loaded from .env
  static String get mediamtxUrl => 
    dotenv.env['MEDIAMTX_URL'] ?? 'http://localhost:8889';

  // WebSocket Endpoints
  static String get threatsWebSocketUrl {
    final wsBase = backendWsUrl.replaceFirst('http://', 'ws://').replaceFirst('https://', 'wss://');
    return '$wsBase/ws/threats';
  }

  // API Endpoints
  static String get authVerifyFirebaseUrl => '$backendUrl/api/v1/auth/verify-firebase';
  static String get authRefreshTokenUrl => '$backendUrl/api/v1/auth/refresh';
  static String get camerasListUrl => '$backendUrl/api/v1/cameras';
  static String cameraStreamUrl(String cameraId) => '$backendUrl/api/v1/streams/$cameraId/url';

  // WHEP Stream URL from MediaMTX
  static String whepStreamUrl(String cameraId) => '$mediamtxUrl/$cameraId';

  /// Debug info: Print current configuration
  static void printConfig() {
    print('=== AppConfig ===');
    print('Backend URL: $backendUrl');
    print('Backend WS URL: $backendWsUrl');
    print('MediaMTX URL: $mediamtxUrl');
    print('Threats WS: $threatsWebSocketUrl');
    print('==================');
  }
}
