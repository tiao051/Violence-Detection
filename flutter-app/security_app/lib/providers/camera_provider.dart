import 'package:flutter/material.dart';
import 'package:security_app/models/camera_model.dart';
import 'package:security_app/services/camera_service.dart';

/// Provider for managing camera-related state.
class CameraProvider with ChangeNotifier {
  final CameraService _cameraService = CameraService();

  List<CameraModel> _cameras = [];
  bool _isLoading = false;
  String? _errorMessage;

  List<CameraModel> get cameras => _cameras;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;

  /// Fetches the list of cameras from the service.
  ///
  /// Skips refetching if cameras are already loaded. Remove this check
  /// if you want to implement pull-to-refresh functionality.
  Future<void> fetchCameras() async {
    if (_cameras.isNotEmpty) return;

    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      _cameras = await _cameraService.getCameras();
    } catch (e) {
      _errorMessage = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
}