import 'package:flutter/material.dart';
import 'package:security_app/models/event_model.dart';
import 'package:security_app/services/event_service.dart';

enum DateFilter { all, today, thisWeek, thisMonth }

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

  // State for date filtering
  DateFilter _dateFilter = DateFilter.all;

  List<EventModel> get events => _events;
  bool get isLoading => _isLoadingEvents;
  String? get errorMessage => _fetchError;
  String? get reportError => _reportError;
  DateFilter get dateFilter => _dateFilter;

  /// Returns true if the given event is currently being reported.
  bool isReporting(String eventId) => _reportingEvents.contains(eventId);

  /// Returns filtered events based on current date filter
  List<EventModel> get filteredEvents {
    final now = DateTime.now();
    switch (_dateFilter) {
      case DateFilter.today:
        return _events
            .where((event) =>
                event.timestamp.isAfter(now.subtract(const Duration(days: 1))))
            .toList();
      case DateFilter.thisWeek:
        return _events
            .where((event) =>
                event.timestamp.isAfter(now.subtract(const Duration(days: 7))))
            .toList();
      case DateFilter.thisMonth:
        return _events
            .where((event) =>
                event.timestamp
                    .isAfter(now.subtract(const Duration(days: 30))))
            .toList();
      case DateFilter.all:
        return _events;
    }
  }

  /// Gets a readable label for the current date filter
  String getFilterLabel() {
    switch (_dateFilter) {
      case DateFilter.today:
        return 'Today';
      case DateFilter.thisWeek:
        return 'This Week';
      case DateFilter.thisMonth:
        return 'This Month';
      case DateFilter.all:
        return 'All Events';
    }
  }

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

  /// Sets the current date filter and triggers rebuild
  void setDateFilter(DateFilter filter) {
    _dateFilter = filter;
    notifyListeners();
  }
}