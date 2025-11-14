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

  /// Whether the event has been viewed by the user.
  final bool isViewed;

  EventModel({
    required this.id,
    required this.cameraName,
    required this.thumbnailUrl,
    required this.videoUrl,
    required this.timestamp,
    this.isViewed = false,
  });

  /// Create a copy of this event with optional field overrides.
  EventModel copyWith({
    String? id,
    String? cameraName,
    String? thumbnailUrl,
    String? videoUrl,
    DateTime? timestamp,
    bool? isViewed,
  }) {
    return EventModel(
      id: id ?? this.id,
      cameraName: cameraName ?? this.cameraName,
      thumbnailUrl: thumbnailUrl ?? this.thumbnailUrl,
      videoUrl: videoUrl ?? this.videoUrl,
      timestamp: timestamp ?? this.timestamp,
      isViewed: isViewed ?? this.isViewed,
    );
  }
}

/// Dummy list of events for initial UI display.
final List<EventModel> dummyEvents = [
  EventModel(
    id: 'evt_001',
    cameraName: 'Front Gate Camera',
    thumbnailUrl: 'https://picsum.photos/seed/event1/120/90',
    videoUrl: 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    timestamp: DateTime.now().subtract(const Duration(minutes: 5)),
    isViewed: false, // Unviewed - will show in badge
  ),
  EventModel(
    id: 'evt_002',
    cameraName: 'Kitchen Camera',
    thumbnailUrl: 'https://picsum.photos/seed/event2/120/90',
    videoUrl: 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
    timestamp: DateTime.now().subtract(const Duration(minutes: 15)),
    isViewed: true, // Viewed - won't count in badge
  ),
  EventModel(
    id: 'evt_003',
    cameraName: 'Back Door Camera',
    thumbnailUrl: 'https://picsum.photos/seed/event3/120/90',
    videoUrl: 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
    timestamp: DateTime.now().subtract(const Duration(hours: 1)),
    isViewed: false, // Unviewed - will show in badge
  ),
  EventModel(
    id: 'evt_004',
    cameraName: 'Front Gate Camera',
    thumbnailUrl: 'https://picsum.photos/seed/event4/120/90',
    videoUrl: 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4',
    timestamp: DateTime.now().subtract(const Duration(hours: 2)),
    isViewed: false, // Unviewed - will show in badge
  ),
  EventModel(
    id: 'evt_005',
    cameraName: 'Parking Lot Camera',
    thumbnailUrl: 'https://picsum.photos/seed/event5/120/90',
    videoUrl: 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4',
    timestamp: DateTime.now().subtract(const Duration(hours: 3)),
    isViewed: true, // Viewed - won't count in badge
  ),
];