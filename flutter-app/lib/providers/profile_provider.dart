import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:security_app/models/profile_model.dart';
import 'package:security_app/services/auth_service.dart';

/// Provider for managing user profile state
class ProfileProvider with ChangeNotifier {
  final FirebaseAuth _firebaseAuth = FirebaseAuth.instance;
  final AuthService _authService = AuthService();

  ProfileModel? _profile;
  bool _isLoading = false;
  String? _errorMessage;

  /// Getters
  ProfileModel? get profile => _profile;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;
  bool get hasProfile => _profile != null;

  /// Load profile from Firebase Auth
  /// Called when app loads to populate profile from current user
  Future<void> loadProfile() async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      final user = _firebaseAuth.currentUser;

      if (user == null) {
        throw Exception('No user logged in');
      }

      print('ProfileProvider: Loading profile for user: ${user.email}');

      // Create profile from Firebase user
      _profile = ProfileModel.fromFirebaseUser(
        user.uid,
        user.email ?? 'unknown@email.com',
        user.displayName,
        user.photoURL,
      );

      print('ProfileProvider: Profile loaded: ${_profile?.displayName}');

      _isLoading = false;
      notifyListeners();
    } catch (e) {
      print('ProfileProvider: Error loading profile: $e');
      _errorMessage = e.toString();
      _isLoading = false;
      notifyListeners();
    }
  }

  /// Update profile display name
  /// In future: can add update profile picture, bio, etc
  Future<void> updateDisplayName(String newName) async {
    if (_profile == null) {
      throw Exception('No profile loaded');
    }

    try {
      final user = _firebaseAuth.currentUser;
      if (user == null) {
        throw Exception('No user logged in');
      }

      // Update Firebase user profile
      await user.updateDisplayName(newName);
      print('ProfileProvider: Updated display name to: $newName');

      // Update local profile
      _profile = _profile!.copyWith(displayName: newName);
      notifyListeners();

      print('ProfileProvider: Profile updated in app');
    } catch (e) {
      print('ProfileProvider: Error updating profile: $e');
      _errorMessage = e.toString();
      notifyListeners();
      rethrow;
    }
  }

  /// Logout user from Firebase and app
  /// Clears profile, token, and signs out from Firebase
  Future<void> logout() async {
    try {
      print('ProfileProvider: Logging out...');

      // Sign out from Firebase
      await _firebaseAuth.signOut();
      print('ProfileProvider: Signed out from Firebase');

      // Sign out from Google
      await _authService.signOutGoogle();
      print('ProfileProvider: Signed out from Google');

      // Clear profile
      _profile = null;
      _errorMessage = null;
      notifyListeners();

      print('ProfileProvider: Profile cleared, logout complete');
    } catch (e) {
      print('ProfileProvider: Error during logout: $e');
      _errorMessage = e.toString();
      notifyListeners();
      rethrow;
    }
  }

  /// Refresh profile data from Firebase
  Future<void> refreshProfile() async {
    await loadProfile();
  }
}
