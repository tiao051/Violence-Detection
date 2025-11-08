/// Model representing an event in the security app.
class EventModel {
  /// Unique identifier for the event.
  final String id;

  /// Name of the camera that captured the event.
  final String cameraName;

  /// URL for the event's thumbnail image.
  final String thumbnailUrl;

  /// URL for the video clip of the event.
  final String videoUrl;

  /// Timestamp when the event occurred.
  final DateTime timestamp;

  EventModel({
    required this.id,
    required this.cameraName,
    required this.thumbnailUrl,
    required this.videoUrl,
    required this.timestamp,
  });
}

/// Dummy list of events for initial UI display.
final List<EventModel> dummyEvents = [
  EventModel(
    id: 'evt_001',
    cameraName: 'Front Gate Camera',
    thumbnailUrl: '',
    videoUrl: 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    timestamp: DateTime.now().subtract(const Duration(minutes: 15)),
  ),
  EventModel(
    id: 'evt_002',
    cameraName: 'Kitchen Camera',
    thumbnailUrl: '',
    videoUrl: 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
    timestamp: DateTime.now().subtract(const Duration(hours: 2)),
  ),
];