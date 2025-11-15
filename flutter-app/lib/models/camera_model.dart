/// Model representing a camera in the security app.
class CameraModel {
  /// Unique identifier for the camera.
  final String id;

  /// Display name of the camera.
  final String name;

  /// URL for the camera's thumbnail image.
  final String thumbnailUrl;

  CameraModel({
    required this.id,
    required this.name,
    required this.thumbnailUrl,
  });
}

/// Dummy list of cameras for initial UI display.
final List<CameraModel> dummyCameras = [
  CameraModel(
    id: "cam-001",
    name: "Front Gate Camera",
    thumbnailUrl: "",
  ),
  CameraModel(
    id: "cam-002",
    name: "Living Room Camera",
    thumbnailUrl: "",
  ),
  CameraModel(
    id: "cam-003",
    name: "Rooftop Camera",
    thumbnailUrl: "",
  ),
  CameraModel(
    id: "cam-004",
    name: "Kitchen Camera",
    thumbnailUrl: "",
  ),
];