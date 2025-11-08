import 'package:security_app/models/event_model.dart';

// Simulated event data for development. Moved outside class to allow
// reuse in other services if needed.
final List<EventModel> dummyEvents = [
  EventModel(
    id: 'evt_001',
    cameraName: 'Front Gate Camera',
    thumbnailUrl: '', // Will use thumbnail images later
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

class EventService {
  
  /// Simulates fetching detected events from backend.
  ///
  /// Uses 800ms delay to mimic network latency.
  /// Returns hardcoded event list until real API is integrated.
  Future<List<EventModel>> getEvents() async {
    await Future.delayed(const Duration(milliseconds: 800));
    print("EventService: Returning simulated event list.");
    return dummyEvents;
  }

  /// Simulates reporting a false detection to backend.
  ///
  /// This would typically send a POST request to mark an event as
  /// incorrectly classified, helping improve the ML model.
  ///
  /// Throws [Exception] if the backend returns an error.
  Future<void> reportEvent(String eventId) async {
    await Future.delayed(const Duration(seconds: 1));
    print("EventService: Report received for event $eventId (simulated)");

    // Uncomment to test error handling:
    // if (eventId == 'evt_001') {
    //   throw Exception("Cannot report (Error 500)");
    // }

    return;
  }
}