import 'package:security_app/models/camera_model.dart'; // Reuse existing model

/// Service layer for camera-related API calls.
class CameraService {
  /// Simulates fetching the list of cameras from backend.
  ///
  /// Uses 1-second delay to mimic network latency.
  /// Returns hardcoded camera list until real API is integrated.
  Future<List<CameraModel>> getCameras() async {
    await Future.delayed(const Duration(seconds: 1));
    print("CameraService: Returning dummy camera list.");
    return dummyCameras;

    // FIXME: Replace with real API call when backend is ready
    // throw Exception("Failed to load camera list (Error 500)");
  }

  /// Simulates fetching the stream URL for a specific camera.
  ///
  /// Returns different test URLs per camera. MP4 is used instead of HLS
  /// because HLS streams often fail on Android emulators and some devices.
  Future<String> getStreamUrl(String cameraId) async {
    await Future.delayed(const Duration(milliseconds: 500));
    print("CameraService: Getting stream URL for camera $cameraId");

    switch (cameraId) {
      case 'cam_001':
        return 'https://cph-p2p-msl.akamaized.net/hls/live/2000341/test/master.m3u8';
      case 'cam_002':
        return 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4';
      case 'cam_003':
        return 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4';
      default:
        return 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4';
    }
  }
}