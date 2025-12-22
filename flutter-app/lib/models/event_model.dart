import 'package:cloud_firestore/cloud_firestore.dart';

/// Model representing an event in the security app.
class EventModel {
  final String id;
  final String userId;
  final String cameraId;
  final String cameraName;
  final String type;
  final String status;
  final String thumbnailUrl;
  final String videoUrl;
  final String imageBase64; // Base64 snapshot for fallback display
  final DateTime timestamp;
  final bool viewed; // Dùng 'viewed' (thay vì 'isViewed') để khớp rule

  EventModel({
    required this.id,
    required this.userId,
    required this.cameraId,
    required this.cameraName,
    required this.type,
    required this.status,
    required this.thumbnailUrl,
    required this.videoUrl,
    required this.imageBase64,
    required this.timestamp,
    required this.viewed,
  });

  /// Factory constructor to create an EventModel from a Firestore document.
  factory EventModel.fromJson(Map<String, dynamic> json, String documentId) {
    final timestamp =
        (json['timestamp'] as Timestamp? ?? Timestamp.now()).toDate();

    return EventModel(
      id: documentId,
      userId: json['userId'] as String? ?? '',
      cameraId: json['cameraId'] as String? ?? '',
      cameraName: json['cameraName'] as String? ?? 'Unknown Camera',
      type: json['type'] as String? ?? 'unknown',
      status: json['status'] as String? ?? 'new',
      thumbnailUrl: json['thumbnailUrl'] as String? ?? '',
      videoUrl: json['videoUrl'] as String? ?? '',
      imageBase64: json['imageBase64'] as String? ?? '',
      timestamp: timestamp,
      viewed: json['viewed'] as bool? ?? false,
    );
  }

  /// Create a copy of this event with optional field overrides.
  /// (ĐÃ THÊM TRỞ LẠI)
  EventModel copyWith({
    String? id,
    String? userId,
    String? cameraId,
    String? cameraName,
    String? type,
    String? status,
    String? thumbnailUrl,
    String? videoUrl,
    String? imageBase64,
    DateTime? timestamp,
    bool? viewed,
  }) {
    return EventModel(
      id: id ?? this.id,
      userId: userId ?? this.userId,
      cameraId: cameraId ?? this.cameraId,
      cameraName: cameraName ?? this.cameraName,
      type: type ?? this.type,
      status: status ?? this.status,
      thumbnailUrl: thumbnailUrl ?? this.thumbnailUrl,
      videoUrl: videoUrl ?? this.videoUrl,
      imageBase64: imageBase64 ?? this.imageBase64,
      timestamp: timestamp ?? this.timestamp,
      viewed: viewed ?? this.viewed,
    );
  }
}
