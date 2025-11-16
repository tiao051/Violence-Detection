/// Model for user profile information
class ProfileModel {
  final String id;
  final String email;
  final String displayName;
  final String? photoUrl;
  final DateTime? createdAt;

  ProfileModel({
    required this.id,
    required this.email,
    required this.displayName,
    this.photoUrl,
    this.createdAt,
  });

  /// Convert ProfileModel to JSON for storage/API
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'email': email,
      'displayName': displayName,
      'photoUrl': photoUrl,
      'createdAt': createdAt?.toIso8601String(),
    };
  }

  /// Create ProfileModel from JSON
  factory ProfileModel.fromJson(Map<String, dynamic> json) {
    return ProfileModel(
      id: json['id'] as String,
      email: json['email'] as String,
      displayName: json['displayName'] as String,
      photoUrl: json['photoUrl'] as String?,
      createdAt: json['createdAt'] != null
          ? DateTime.parse(json['createdAt'] as String)
          : null,
    );
  }

  /// Create ProfileModel from Firebase User
  factory ProfileModel.fromFirebaseUser(String id, String email, String? displayName, String? photoUrl) {
    return ProfileModel(
      id: id,
      email: email,
      displayName: displayName ?? 'User',
      photoUrl: photoUrl,
      createdAt: DateTime.now(),
    );
  }

  /// Copy with method for creating modified copies
  ProfileModel copyWith({
    String? id,
    String? email,
    String? displayName,
    String? photoUrl,
    DateTime? createdAt,
  }) {
    return ProfileModel(
      id: id ?? this.id,
      email: email ?? this.email,
      displayName: displayName ?? this.displayName,
      photoUrl: photoUrl ?? this.photoUrl,
      createdAt: createdAt ?? this.createdAt,
    );
  }

  @override
  String toString() =>
      'ProfileModel(id: $id, email: $email, displayName: $displayName, photoUrl: $photoUrl, createdAt: $createdAt)';
}
