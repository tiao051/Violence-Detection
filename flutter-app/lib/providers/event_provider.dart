import 'package:flutter/material.dart';
import 'package:security_app/models/event_model.dart';
import 'package:security_app/services/event_service.dart';

enum DateFilter { all, today, thisWeek, thisMonth }

class EventProvider with ChangeNotifier {
  final EventService _eventService = EventService();

  List<EventModel> _events = [];
  bool _isLoadingEvents = false;
  String? _fetchError;

  final Set<String> _reportingEvents = {};
  String? _reportError;
  
  // State mới để theo dõi việc "marking as viewed"
  final Set<String> _viewingEvents = {};
  String? _viewError;

  DateFilter _dateFilter = DateFilter.all;

  List<EventModel> get events => _events;
  bool get isLoading => _isLoadingEvents;
  String? get errorMessage => _fetchError;
  String? get reportError => _reportError;
  String? get viewError => _viewError;
  DateFilter get dateFilter => _dateFilter;

  bool isReporting(String eventId) => _reportingEvents.contains(eventId);
  bool isMarkingViewed(String eventId) => _viewingEvents.contains(eventId);

  /// SỬA LỖI 1: Đổi 'isViewed' thành 'viewed'
  int get unviewedCount => _events.where((event) => !event.viewed).length;

  /// SỬA LỖI 2: Đổi 'isViewed' thành 'viewed'
  List<EventModel> get unviewedEvents =>
      _events.where((event) => !event.viewed).toList();

  /// (filteredEvents không đổi, vì nó dùng 'timestamp')
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

  String getFilterLabel() {
    switch (_dateFilter) {
      case DateFilter.today: return 'Today';
      case DateFilter.thisWeek: return 'This Week';
      case DateFilter.thisMonth: return 'This Month';
      case DateFilter.all: return 'All Events';
    }
  }

  /// SỬA LỖI 3: Đơn giản hóa logic fetch (state 'viewed' đã đến từ Firestore)
  Future<void> fetchEvents() async {
    _isLoadingEvents = true;
    _fetchError = null;
    notifyListeners();

    try {
      // getEvents() mới sẽ lấy từ Firestore
      final newEvents = await _eventService.getEvents();
      _events = newEvents;
    } catch (e) {
      _fetchError = e.toString().replaceFirst("Exception: ", "");
    } finally {
      _isLoadingEvents = false;
      notifyListeners();
    }
  }

  /// SỬA LỖI 4: Đổi tên hàm service
  Future<bool> reportEvent(String eventId) async {
    _reportingEvents.add(eventId);
    _reportError = null;
    notifyListeners();

    try {
      // Gọi hàm service mới 'reportEventAsFalse'
      await _eventService.reportEventAsFalse(eventId);

      // Cập nhật UI ngay lập tức (Optimistic Update)
      final eventIndex = _events.indexWhere((e) => e.id == eventId);
      if (eventIndex != -1) {
        _events = [
          ..._events.sublist(0, eventIndex),
          _events[eventIndex].copyWith(status: 'reported_false'),
          ..._events.sublist(eventIndex + 1),
        ];
      }
      return true;
    } catch (e) {
      _reportError = e.toString().replaceFirst("Exception: ", "");
      return false;
    } finally {
      _reportingEvents.remove(eventId);
      notifyListeners();
    }
  }

  void clearCache() {
    _events = [];
    _isLoadingEvents = false;
    _fetchError = null;
    _reportingEvents.clear();
    _reportError = null;
    notifyListeners();
  }

  /// SỬA LỖI 5: Đơn giản hóa logic refresh (giống hệt fetchEvents)
  Future<void> refreshEvents() async {
    _fetchError = null;
    _isLoadingEvents = true;
    notifyListeners();

    try {
      final newEvents = await _eventService.getEvents();
      _events = newEvents;
      _fetchError = null;
    } catch (e) {
      _fetchError = e.toString().replaceFirst("Exception: ", "");
    } finally {
      _isLoadingEvents = false;
      notifyListeners();
    }
  }

  void setDateFilter(DateFilter filter) {
    _dateFilter = filter;
    notifyListeners();
  }

  /// SỬA LỖI 6: Nâng cấp hàm 'markEventAsViewed' để gọi service
  /// (Hàm cũ chỉ thay đổi state local)
  Future<void> markEventAsViewedInDb(String eventId) async {
    // Tìm event trước
    final eventIndex = _events.indexWhere((event) => event.id == eventId);
    if (eventIndex == -1) return; // Không tìm thấy
    if (_events[eventIndex].viewed) return; // Đã xem rồi, không cần gọi

    _viewingEvents.add(eventId);
    _viewError = null;
    notifyListeners();

    // 1. Cập nhật UI ngay lập tức (Optimistic Update)
    _events = [
      ..._events.sublist(0, eventIndex),
      _events[eventIndex].copyWith(viewed: true), // Sửa 'isViewed' thành 'viewed'
      ..._events.sublist(eventIndex + 1),
    ];
    notifyListeners();

    // 2. Gọi service để cập nhật Firestore
    try {
      await _eventService.markEventAsViewed(eventId);
    } catch (e) {
      _viewError = e.toString().replaceFirst("Exception: ", "");
      // 3. Nếu thất bại, khôi phục lại (Revert)
      _events = [
        ..._events.sublist(0, eventIndex),
        _events[eventIndex].copyWith(viewed: false), // Khôi phục
        ..._events.sublist(eventIndex + 1),
      ];
      notifyListeners();
    } finally {
      _viewingEvents.remove(eventId);
      notifyListeners();
    }
  }
}