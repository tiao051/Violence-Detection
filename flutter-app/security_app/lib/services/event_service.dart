// [File: lib/services/event_service.dart]
import 'package:security_app/models/event_model.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class EventService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  
  /// Fetches detected events from Firestore for the current user.
  Future<List<EventModel>> getEvents() async {
    print("EventService: Fetching events from Firestore...");
    
    final User? user = _auth.currentUser;
    if (user == null) {
      throw Exception('User not logged in. Cannot fetch events.');
    }

    try {
      // This query tests the Security Rule:
      // allow read: if request.auth.uid == resource.data.userId;
      final querySnapshot = await _firestore
          .collection('events')
          .where('userId', isEqualTo: user.uid)
          .orderBy('timestamp', descending: true)
          .get();
          
      final events = querySnapshot.docs.map((doc) {
        return EventModel.fromJson(doc.data(), doc.id);
      }).toList();

      print("EventService: Found ${events.length} events for user.");
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
  /// (ĐÃ XÓA CODE DEBUG)
  Future<void> markEventAsViewed(String eventId) async {
    print("EventService: Marking event $eventId as viewed (Firestore)");
    
    final User? user = _auth.currentUser;
    if (user == null) {
      throw Exception('User not logged in. Cannot update.');
    }

    try {
      // Attempt the update
      await _firestore.collection('events').doc(eventId).update({
        'viewed': true, // <--- CỐ TÌNH GÕ SAI
      });
      
      print("EventService: Event $eventId 'viewed' status updated.");
    } catch (e) {
      print("EventService: Error marking event as viewed: $e");
      throw Exception("Failed to mark event as viewed: $e");
    }
  }
}