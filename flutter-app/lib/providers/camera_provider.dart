import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:security_app/models/camera_model.dart';
import 'package:security_app/services/camera_service.dart';
import 'package:security_app/providers/auth_provider.dart';

/// Provider for managing camera-related state.
class CameraProvider with ChangeNotifier {
  final CameraService _cameraService = CameraService();

  List<CameraModel> _cameras = [];
  bool _isLoading = false;
  String? _errorMessage;
  String _searchQuery = '';

  List<CameraModel> get cameras => _cameras;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;
  String get searchQuery => _searchQuery;

  /// Returns filtered cameras based on current search query
  /// Filters by camera name and ID (case-insensitive)
  List<CameraModel> get filteredCameras {
    if (_searchQuery.isEmpty) {
      return _cameras;
    }
    final query = _searchQuery.toLowerCase();
    return _cameras
        .where((camera) =>
            camera.name.toLowerCase().contains(query) ||
            camera.id.toLowerCase().contains(query))
        .toList();
  }

  /// Fetches the list of cameras from the backend service.
  ///
  /// Requires access_token from AuthProvider. Skips refetching if cameras
  /// are already loaded. Remove this check if you want pull-to-refresh.
  Future<void> fetchCameras({required String accessToken}) async {
    if (_cameras.isNotEmpty) return;

    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      _cameras = await _cameraService.getCameras(accessToken: accessToken);
    } catch (e) {
      _errorMessage = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  /// Clears all cached camera data.
  ///
  /// Called when user logs out to prevent data leaks between sessions.
  void clearCache() {
    _cameras = [];
    _isLoading = false;
    _errorMessage = null;
    notifyListeners();
  }

  /// Refreshes the camera list by clearing cache and fetching fresh data.
  ///
  /// Called by pull-to-refresh gesture. Returns a Future that completes
  /// when the fetch operation is done (success or failure).
  Future<void> refreshCameras({required String accessToken}) async {
    // Clear cache to force fresh fetch
    _cameras = [];
    _errorMessage = null;
    _isLoading = true;
    notifyListeners();

    try {
      _cameras = await _cameraService.getCameras(accessToken: accessToken);
      _errorMessage = null;
    } catch (e) {
      _errorMessage = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  /// Updates the search query and filters cameras accordingly.
  ///
  /// Filtering happens instantly without API calls.
  void setSearchQuery(String query) {
    _searchQuery = query;
    notifyListeners();
  }

  /// Clears the current search query, showing all cameras.
  void clearSearch() {
    _searchQuery = '';
    notifyListeners();
  }
}