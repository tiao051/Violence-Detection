import 'package:cloud_firestore/cloud_firestore.dart';

/// Model representing a camera in the security app.
class CameraModel {
  /// Unique identifier for the camera (document ID).
  final String id;

  /// Display name of the camera.
  final String name;
  
  /// Location of the camera (e.g., "Front Gate").
  final String location;

  /// The RTSP/HTTP stream URL.
  final String streamUrl;
  
  /// List of user UIDs allowed to view this camera.
  final List<String> assignedUsers;

  CameraModel({
    required this.id,
    required this.name,
    required this.location,
    required this.streamUrl,
    required this.assignedUsers,
  });

  /// Factory constructor to create a CameraModel from a Firestore document.
  factory CameraModel.fromJson(Map<String, dynamic> json, String documentId) {
    // Convert 'assignedUsers' from List<dynamic> to List<String>
    final assignedUsersDynamic = json['assignedUsers'] as List<dynamic>? ?? [];
    final assignedUsersString = assignedUsersDynamic.map((item) => item.toString()).toList();

    return CameraModel(
      id: documentId,
      name: json['name'] as String? ?? 'Unnamed Camera',
      location: json['location'] as String? ?? 'Unknown Location',
      streamUrl: json['streamUrl'] as String? ?? '',
      assignedUsers: assignedUsersString,
    );
  }
}