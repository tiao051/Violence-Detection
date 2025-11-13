import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:security_app/services/auth_service.dart';

/// Provider for managing authentication state.
class AuthProvider with ChangeNotifier {
  final AuthService _authService = AuthService();
  final SharedPreferences _prefs;

  String? _token;
  bool _isLoading = false;
  bool _isLoadingGoogle = false;
  String? _errorMessage;

  /// Constructor that restores authentication state from previous session.
  ///
  /// Checks SharedPreferences for saved token to maintain login state
  /// across app restarts.
  AuthProvider(this._prefs) {
    _token = _prefs.getString('token');
    if (_token != null) {
      print("AuthProvider: Token found, user is logged in.");
    }
  }

  bool get isLoggedIn => _token != null;
  bool get isLoading => _isLoading;
  bool get isLoadingGoogle => _isLoadingGoogle;
  String? get errorMessage => _errorMessage;

  /// Logs in the user with email and password.
  ///
  /// Persists the auth token to SharedPreferences so users remain logged in
  /// after closing the app. UI listeners are notified at each state change.
  Future<void> login(String email, String password) async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      final token = await _authService.login(email, password);
      _token = token;
      await _prefs.setString('token', token);

      _isLoading = false;
      notifyListeners();
    } catch (e) {
      _token = null;
      _errorMessage = e.toString();
      _isLoading = false;
      notifyListeners();
    }
  }

  /// Initiates Google Sign-In flow via AuthService.
  ///
  /// Sets _isLoadingGoogle flag to show spinner on Google button only.
  /// Persists token to SharedPreferences on success.
  Future<void> signInWithGoogleProvider() async {
    _isLoadingGoogle = true;
    _errorMessage = null;
    notifyListeners();

    try {
      final token = await _authService.signInWithGoogle();
      _token = token;
      await _prefs.setString('token', token);

      _isLoadingGoogle = false;
      notifyListeners();
    } catch (e) {
      _token = null;
      _errorMessage = e.toString();
      _isLoadingGoogle = false;
      notifyListeners();
    }
  }

  /// Logs out the current user and clears persisted session.
  ///
  /// Also signs out from Google to forget account selection.
  Future<void> logout() async {
    _token = null;
    await _prefs.remove('token');
    await _authService.signOutGoogle();
    notifyListeners();
  }
}