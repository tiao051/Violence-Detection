import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:security_app/services/auth_service.dart';
import 'package:security_app/models/auth_model.dart';

/// Provider for managing authentication state.
class AuthProvider with ChangeNotifier {
  final AuthService _authService = AuthService();
  final SharedPreferences _prefs;

  String? _accessToken;      // JWT access token for API calls
  String? _refreshToken;     // JWT refresh token for renewal
  bool _isLoading = false;
  bool _isLoadingGoogle = false;
  bool _isLoadingSignUp = false;
  bool _isChangingPassword = false;
  String? _errorMessage;
  AuthModel? _user;

  /// Constructor that restores authentication state from previous session.
  ///
  /// Checks SharedPreferences for saved tokens to maintain login state
  /// across app restarts.
  AuthProvider(this._prefs) {
    _accessToken = _prefs.getString('access_token');
    _refreshToken = _prefs.getString('refresh_token');
    if (_accessToken != null) {
      print("AuthProvider: Access token found, user is logged in.");
    }
  }

  bool get isLoggedIn => _accessToken != null;
  bool get isLoading => _isLoading;
  bool get isLoadingGoogle => _isLoadingGoogle;
  bool get isLoadingSignUp => _isLoadingSignUp;
  bool get isChangingPassword => _isChangingPassword;
  String? get errorMessage => _errorMessage;
  AuthModel? get user => _user;
  String? get accessToken => _accessToken;  // Expose token for API calls
  String? get refreshToken => _refreshToken;

  /// Initiates Google Sign-In via Firebase Authentication.
  ///
  /// Flow:
  /// 1. User taps "Sign in with Google" button
  /// 2. GoogleSignIn SDK shows Google account picker
  /// 3. User selects their Google account (@gmail.com, @company.com, etc)
  /// 4. AuthService exchanges Google credential for Firebase credential
  /// 5. Firebase verifies credential and creates/updates user
  /// 6. AuthService retrieves Firebase ID token (JWT)
  /// 7. Send Firebase ID token to backend POST /api/v1/auth/verify-firebase
  /// 8. Backend verifies token with Firebase and returns JWT (access + refresh)
  /// 9. App saves both access_token and refresh_token to SharedPreferences
  ///
  /// Sets _isLoadingGoogle flag to show spinner on Google button only.
  /// Persists tokens to SharedPreferences on success.
  Future<void> signInWithGoogleProvider() async {
    _isLoadingGoogle = true;
    _errorMessage = null;
    notifyListeners();

    try {
      // Get Firebase ID token from AuthService
      final firebaseIdToken = await _authService.signInWithGoogle();
      
      print('AuthProvider: Firebase ID token received (length: ${firebaseIdToken.length})');
      
      // Send Firebase ID token to backend for JWT exchange
      final tokens = await _authService.verifyFirebaseToken(firebaseIdToken);
      
      _accessToken = tokens['access_token'];
      _refreshToken = tokens['refresh_token'];
      
      // Save tokens to SharedPreferences
      await _prefs.setString('access_token', _accessToken!);
      await _prefs.setString('refresh_token', _refreshToken!);

      print('AuthProvider: JWT tokens saved to SharedPreferences');
      
      _isLoadingGoogle = false;
      notifyListeners();
      
      print('AuthProvider: Firebase Sign-In complete, isLoggedIn: $isLoggedIn');
    } catch (e) {
      print('AuthProvider: Firebase Sign-In failed: $e');
      _accessToken = null;
      _refreshToken = null;
      _errorMessage = e.toString();
      _isLoadingGoogle = false;
      notifyListeners();
    }
  }

  /// Logs out the current user and clears persisted session.
  ///
  /// Also signs out from Google to forget account selection.
  Future<void> logout() async {
    _accessToken = null;
    _refreshToken = null;
    _user = null;
    await _prefs.remove('access_token');
    await _prefs.remove('refresh_token');
    await _authService.signOutGoogle();
    notifyListeners();
  }

  /// Signs up a new user with email, password, and display name
  ///
  /// Flow:
  /// 1. Call AuthService.signUpWithEmail()
  /// 2. Get Firebase ID token
  /// 3. Exchange for JWT tokens via backend /api/v1/auth/verify-firebase
  /// 4. Save tokens to SharedPreferences
  ///
  /// Throws error if email already exists or validation fails
  Future<void> signUp({
    required String email,
    required String password,
    required String displayName,
  }) async {
    _isLoadingSignUp = true;
    _errorMessage = null;
    notifyListeners();

    try {
      print('AuthProvider: Starting sign up for email: $email');

      // Call auth service to create user and get Firebase ID token
      final firebaseIdToken = await _authService.signUpWithEmail(
        email: email,
        password: password,
        displayName: displayName,
      );

      print('AuthProvider: Sign up successful, exchanging for JWT');

      // Exchange Firebase ID token for JWT
      final tokens = await _authService.verifyFirebaseToken(firebaseIdToken);
      
      _accessToken = tokens['access_token'];
      _refreshToken = tokens['refresh_token'];

      // Save tokens
      await _prefs.setString('access_token', _accessToken!);
      await _prefs.setString('refresh_token', _refreshToken!);

      print('AuthProvider: JWT tokens saved');

      _isLoadingSignUp = false;
      notifyListeners();
    } catch (e) {
      print('AuthProvider: Sign up error: $e');
      _accessToken = null;
      _refreshToken = null;
      _errorMessage = e.toString();
      _isLoadingSignUp = false;
      notifyListeners();
    }
  }

  /// Logs in user with email and password
  ///
  /// Exchanges Firebase ID token for JWT (access + refresh)
  /// Persists tokens to SharedPreferences so users remain logged in
  /// after closing the app.
  Future<void> loginWithEmail({
    required String email,
    required String password,
  }) async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      print('AuthProvider: Starting login with email: $email');

      // Call auth service for Firebase login
      final firebaseIdToken = await _authService.loginWithEmail(
        email: email,
        password: password,
      );

      print('AuthProvider: Firebase login successful, exchanging for JWT');

      // Exchange Firebase ID token for JWT
      final tokens = await _authService.verifyFirebaseToken(firebaseIdToken);
      
      _accessToken = tokens['access_token'];
      _refreshToken = tokens['refresh_token'];

      // Save tokens
      await _prefs.setString('access_token', _accessToken!);
      await _prefs.setString('refresh_token', _refreshToken!);

      print('AuthProvider: JWT tokens saved');

      _isLoading = false;
      notifyListeners();
    } catch (e) {
      print('AuthProvider: Login error: $e');
      _accessToken = null;
      _refreshToken = null;
      _errorMessage = e.toString();
      _isLoading = false;
      notifyListeners();
      rethrow;
    }
  }

  /// Changes password for the current user
  ///
  /// Requires current password and new password
  /// Shows error if current password is incorrect
  Future<void> changePassword({
    required String currentPassword,
    required String newPassword,
  }) async {
    _isLoading = true;
    _isChangingPassword = true; // Set flag to prevent redirect
    _errorMessage = null;
    notifyListeners();

    try {
      print('AuthProvider: Starting password change');

      await _authService.changePassword(
        currentPassword: currentPassword,
        newPassword: newPassword,
      );

      print('AuthProvider: Password changed successfully');

      _isLoading = false;
      _isChangingPassword = false; // Clear flag
      notifyListeners();
    } catch (e) {
      print('AuthProvider: Change password error: $e');
      _errorMessage = e.toString();
      _isLoading = false;
      _isChangingPassword = false; // Clear flag even on error
      notifyListeners();
      rethrow; // Re-throw to let dialog handle the error message
    }
  }

  /// Updates the display name of the current user
  ///
  /// Updates both Firebase Auth and Firestore
  /// Refreshes local user profile after update
  Future<void> updateProfile({
    required String displayName,
  }) async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      print('AuthProvider: Starting profile update');

      await _authService.updateDisplayName(displayName);

      print('AuthProvider: Display name updated');

      // Refresh user profile from Firestore
      final updatedUser = await _authService.getUserProfile();
      _user = updatedUser;

      print('AuthProvider: Profile refreshed');

      _isLoading = false;
      notifyListeners();
    } catch (e) {
      print('AuthProvider: Update profile error: $e');
      _errorMessage = e.toString();
      _isLoading = false;
      notifyListeners();
      rethrow;
    }
  }

  /// Checks if an email is already registered
  ///
  /// Returns true if email exists, false otherwise
  /// Non-blocking - does not set loading state
  Future<bool> checkEmailExists(String email) async {
    try {
      print('AuthProvider: Checking if email exists: $email');
      final exists = await _authService.emailExists(email);
      print('AuthProvider: Email exists result: $exists');
      return exists;
    } catch (e) {
      print('AuthProvider: Error checking email: $e');
      return false;
    }
  }

  /// Clears the current error message
  ///
  /// Used to reset error state after displaying to user
  void clearError() {
    _errorMessage = null;
    notifyListeners();
  }

  /// Sends a password reset email
  ///
  /// Sets loading state and clears error message
  /// Shows error if email not found or other Firebase errors
  Future<void> sendPasswordResetEmail(String email) async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      print('AuthProvider: Sending password reset email for: $email');
      await _authService.sendPasswordResetEmail(email);
      print('AuthProvider: Password reset email sent successfully');

      _isLoading = false;
      notifyListeners();
    } catch (e) {
      print('AuthProvider: Send password reset email error: $e');
      _errorMessage = e.toString();
      _isLoading = false;
      notifyListeners();
      rethrow;
    }
  }
}
