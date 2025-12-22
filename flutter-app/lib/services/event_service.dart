// [File: lib/services/event_service.dart]
import 'package:security_app/models/event_model.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class EventService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;

  /// Fetches detected events from Firestore for the current user.
  /// Limited to 100 most recent events to prevent OOM.
  Future<List<EventModel>> getEvents() async {
    final User? user = _auth.currentUser;
    if (user == null) {
      throw Exception('User not logged in. Cannot fetch events.');
    }

    try {
      // Limit to 100 events to prevent OutOfMemoryError
      final querySnapshot = await _firestore
          .collection('events')
          .where('userId', isEqualTo: user.uid)
          .orderBy('timestamp', descending: true)
          .limit(100) // IMPORTANT: Limit to prevent OOM
          .get();

      final events = querySnapshot.docs.map((doc) {
        // Remove imageBase64 from list to save memory
        final data = Map<String, dynamic>.from(doc.data());
        data.remove('imageBase64');
        return EventModel.fromJson(data, doc.id);
      }).toList();

      print("EventService: Loaded ${events.length} events");
      return events;
    } catch (e) {
      print("EventService: Error fetching events: $e");
      throw Exception("Failed to load events: $e");
    }
  }

  /// TEST RULE: Updates an event status (e.g., false report)
  /// This will test: hasOnly(['status', 'viewed'])
  Future<void> reportEventAsFalse(String eventId) async {
    print("EventService: Reporting event $eventId as false (Firestore)");

    try {
      await _firestore.collection('events').doc(eventId).update({
        'status': 'reported_false',
      });
      print("EventService: Event $eventId status updated.");
    } catch (e) {
      print("EventService: Error reporting event: $e");
      // This will fail if the rule is not set correctly
      throw Exception("Failed to report event: $e");
    }
  }

  /// TEST RULE: Updates an event's 'viewed' status
  Future<void> markEventAsViewed(String eventId) async {
    print("EventService: Marking event $eventId as viewed (Firestore)");

    final User? user = _auth.currentUser;
    if (user == null) {
      throw Exception('User not logged in. Cannot update.');
    }

    try {
      // Attempt the update
      await _firestore.collection('events').doc(eventId).update({
        'viewed': true,
      });

      print("EventService: Event $eventId 'viewed' status updated.");
    } catch (e) {
      print("EventService: Error marking event as viewed: $e");
      throw Exception("Failed to mark event as viewed: $e");
    }
  }
}
