/// Model for unified user authentication and profile data
class AuthModel {
  final String uid;
  final String email;
  final String displayName;
  final String? photoUrl;
  final String authProvider; // 'email' or 'google'
  final DateTime createdAt;
  final DateTime? updatedAt;

  AuthModel({
    required this.uid,
    required this.email,
    required this.displayName,
    this.photoUrl,
    required this.authProvider,
    required this.createdAt,
    this.updatedAt,
  });

  /// Convert AuthModel to JSON for Firestore storage
  Map<String, dynamic> toJson() {
    return {
      'uid': uid,
      'email': email,
      'displayName': displayName,
      'photoUrl': photoUrl,
      'authProvider': authProvider,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt?.toIso8601String(),
    };
  }

  /// Create AuthModel from JSON (e.g., from Firestore)
  factory AuthModel.fromJson(Map<String, dynamic> json) {
    return AuthModel(
      uid: json['uid'] as String,
      email: json['email'] as String,
      displayName: json['displayName'] as String,
      photoUrl: json['photoUrl'] as String?,
      authProvider: json['authProvider'] as String,
      createdAt: DateTime.parse(json['createdAt'] as String),
      updatedAt: json['updatedAt'] != null
          ? DateTime.parse(json['updatedAt'] as String)
          : null,
    );
  }

  /// Create AuthModel from Firebase User (for email signup)
  factory AuthModel.fromFirebaseUserEmail({
    required String uid,
    required String email,
    required String displayName,
    String? photoUrl,
  }) {
    return AuthModel(
      uid: uid,
      email: email,
      displayName: displayName,
      photoUrl: photoUrl,
      authProvider: 'email',
      createdAt: DateTime.now(),
      updatedAt: DateTime.now(),
    );
  }

  /// Create AuthModel from Firebase User (for Google signup)
  factory AuthModel.fromFirebaseUserGoogle({
    required String uid,
    required String email,
    required String? displayName,
    required String? photoUrl,
  }) {
    return AuthModel(
      uid: uid,
      email: email,
      displayName: displayName ?? 'User',
      photoUrl: photoUrl,
      authProvider: 'google',
      createdAt: DateTime.now(),
      updatedAt: DateTime.now(),
    );
  }

  /// Create a copy with optional field updates
  AuthModel copyWith({
    String? uid,
    String? email,
    String? displayName,
    String? photoUrl,
    String? authProvider,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return AuthModel(
      uid: uid ?? this.uid,
      email: email ?? this.email,
      displayName: displayName ?? this.displayName,
      photoUrl: photoUrl ?? this.photoUrl,
      authProvider: authProvider ?? this.authProvider,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  @override
  String toString() =>
      'AuthModel(uid: $uid, email: $email, displayName: $displayName, photoUrl: $photoUrl, authProvider: $authProvider, createdAt: $createdAt, updatedAt: $updatedAt)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is AuthModel &&
          runtimeType == other.runtimeType &&
          uid == other.uid &&
          email == other.email &&
          displayName == other.displayName &&
          photoUrl == other.photoUrl &&
          authProvider == other.authProvider &&
          createdAt == other.createdAt &&
          updatedAt == other.updatedAt;

  @override
  int get hashCode =>
      uid.hashCode ^
      email.hashCode ^
      displayName.hashCode ^
      photoUrl.hashCode ^
      authProvider.hashCode ^
      createdAt.hashCode ^
      updatedAt.hashCode;
}
