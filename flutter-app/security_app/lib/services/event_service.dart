import 'package:security_app/models/event_model.dart';

class EventService {
  
  /// Simulates fetching detected events from backend.
  ///
  /// Uses 800ms delay to mimic network latency.
  /// Returns hardcoded event list (from event_model.dart) until real API is integrated.
  Future<List<EventModel>> getEvents() async {
    await Future.delayed(const Duration(milliseconds: 800));
    print("EventService: Returning simulated event list with ${dummyEvents.length} events.");
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