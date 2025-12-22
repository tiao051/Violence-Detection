import 'package:security_app/models/camera_model.dart';
import 'package:security_app/config/app_config.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

/// Service layer for camera-related API calls.
class CameraService {
  final FirebaseAuth _auth = FirebaseAuth.instance;

  /// Fetches the list of cameras from backend API
  ///
  /// Calls GET /api/v1/cameras with JWT token from SharedPreferences.
  /// The backend returns cameras owned by the authenticated user.
  ///
  /// Requires:
  /// - JWT access token to be available via AuthProvider
  /// - Backend to be running and accessible
  ///
  /// Returns: List of cameras owned by user
  /// Throws: Exception if API call fails or token is missing
  Future<List<CameraModel>> getCameras({required String accessToken}) async {
    if (accessToken.isEmpty) {
      throw Exception('Access token is missing. User not authenticated.');
    }

    try {
      final uri = Uri.parse(AppConfig.camerasListUrl);

      final response = await http.get(
        uri,
        headers: {
          'Authorization': 'Bearer $accessToken',
          'Content-Type': 'application/json',
        },
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);

        final cameras = data.map((cam) {
          return CameraModel(
            id: cam['id'] ?? '',
            name: cam['name'] ?? 'Unknown',
            location: cam['location'] ?? '',
            streamUrl: cam['stream_url'] ?? '',
          );
        }).toList();

        return cameras;
      } else if (response.statusCode == 401) {
        throw Exception('Unauthorized: Invalid or expired token');
      } else {
        final errorData = jsonDecode(response.body);
        final errorMsg = errorData['detail'] ?? 'Failed to load cameras';
        throw Exception(errorMsg);
      }
    } catch (e) {
      print("CameraService: Error fetching cameras: $e");
      throw Exception("Failed to load camera list: $e");
    }
  }

  /// Fetches the stream URL for a specific camera from backend
  ///
  /// Calls GET /api/v1/streams/{camera_id}/url with JWT token.
  /// Returns authorized WebRTC stream URL.
  ///
  /// Returns: Stream URL string
  /// Throws: Exception if camera not found or user not authorized
  Future<String> getStreamUrl(
    String cameraId, {
    required String accessToken,
  }) async {
    if (accessToken.isEmpty) {
      throw Exception('Access token is missing. User not authenticated.');
    }

    try {
      final uri = Uri.parse(AppConfig.cameraStreamUrl(cameraId));

      final response = await http.get(
        uri,
        headers: {
          'Authorization': 'Bearer $accessToken',
          'Content-Type': 'application/json',
        },
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        if (data['stream_url'] == null) {
          throw Exception('Invalid response: missing stream_url');
        }

        return data['stream_url'];
      } else if (response.statusCode == 403) {
        throw Exception('Not authorized to access this camera');
      } else if (response.statusCode == 404) {
        throw Exception('Camera not found');
      } else if (response.statusCode == 401) {
        throw Exception('Unauthorized: Invalid or expired token');
      } else {
        final errorData = jsonDecode(response.body);
        final errorMsg = errorData['detail'] ?? 'Failed to get stream URL';
        throw Exception(errorMsg);
      }
    } catch (e) {
      print("CameraService: Error getting stream URL: $e");
      throw Exception("Failed to get stream: $e");
    }
  }
}
