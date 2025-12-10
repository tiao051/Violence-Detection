import 'package:cloud_firestore/cloud_firestore.dart';

/// Model representing a camera in the security app.
class CameraModel {
  /// Unique identifier for the camera.
  final String id;

  /// Display name of the camera.
  final String name;
  
  /// Location of the camera (e.g., "Front Gate").
  final String location;

  /// The WebRTC stream URL from backend.
  final String streamUrl;

  CameraModel({
    required this.id,
    required this.name,
    required this.location,
    required this.streamUrl,
  });
}
