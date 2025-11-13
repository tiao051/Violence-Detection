import 'package:flutter/material.dart';
import 'package:security_app/models/event_model.dart';
import 'package:security_app/services/event_service.dart';

class EventProvider with ChangeNotifier {
  final EventService _eventService = EventService();

  // State for fetching events
  List<EventModel> _events = [];
  bool _isLoadingEvents = false;
  String? _fetchError;

  // State for reporting events
  // Using a Set allows tracking multiple report operations simultaneously,
  // enabling individual button loading states in the UI
  final Set<String> _reportingEvents = {};
  String? _reportError;

  List<EventModel> get events => _events;
  bool get isLoading => _isLoadingEvents;
  String? get errorMessage => _fetchError;
  String? get reportError => _reportError;

  /// Returns true if the given event is currently being reported.
  bool isReporting(String eventId) => _reportingEvents.contains(eventId);

  /// Fetches the list of detected events from the service.
  Future<void> fetchEvents() async {
    _isLoadingEvents = true;
    _fetchError = null;
    notifyListeners();

    try {
      _events = await _eventService.getEvents();
    } catch (e) {
      _fetchError = e.toString();
    } finally {
      _isLoadingEvents = false;
      notifyListeners();
    }
  }

  /// Reports a false detection for the given event ID.
  ///
  /// Tracks reporting state per event ID to enable individual button
  /// loading states. Returns true on success, false on failure.
  Future<bool> reportEvent(String eventId) async {
    _reportingEvents.add(eventId);
    _reportError = null;
    notifyListeners();

    try {
      await _eventService.reportEvent(eventId);
      return true;
    } catch (e) {
      _reportError = e.toString();
      return false;
    } finally {
      _reportingEvents.remove(eventId);
      notifyListeners();
    }
  }

  /// Clears all cached event data.
  ///
  /// Called when user logs out to prevent data leaks between sessions.
  void clearCache() {
    _events = [];
    _isLoadingEvents = false;
    _fetchError = null;
    _reportingEvents.clear();
    _reportError = null;
    notifyListeners();
  }

  /// Refreshes the event list by clearing cache and fetching fresh data.
  ///
  /// Called by pull-to-refresh gesture. Returns a Future that completes
  /// when the fetch operation is done (success or failure).
  Future<void> refreshEvents() async {
    // Clear cache to force fresh fetch
    _events = [];
    _fetchError = null;
    _isLoadingEvents = true;
    notifyListeners();

    try {
      _events = await _eventService.getEvents();
      _fetchError = null;
    } catch (e) {
      _fetchError = e.toString();
    } finally {
      _isLoadingEvents = false;
      notifyListeners();
    }
  }
}