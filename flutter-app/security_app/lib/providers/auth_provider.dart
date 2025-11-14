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

  /// Initiates Google Sign-In via Firebase Authentication.
  ///
  /// Real Firebase Sign-In flow:
  /// 1. User taps "Sign in with Google" button
  /// 2. GoogleSignIn SDK shows Google account picker
  /// 3. User selects their Google account (@gmail.com, @company.com, etc)
  /// 4. AuthService exchanges Google credential for Firebase credential
  /// 5. Firebase verifies credential and creates/updates user
  /// 6. AuthService retrieves Firebase ID token (JWT)
  /// 7. ID token is saved locally to SharedPreferences
  /// 8. TODO: When backend is ready, send ID token to POST /auth/firebase-verify
  /// 9. Backend verifies token with Firebase and returns app JWT
  /// 10. App saves app JWT instead of Firebase ID token
  ///
  /// Sets _isLoadingGoogle flag to show spinner on Google button only.
  /// Persists token to SharedPreferences on success.
  /// 
  /// Future integration:
  /// - Replace Firebase ID token with real JWT from backend
  /// - Handle JWT with short expiration (backend decides)
  /// - Refresh JWT when expired
  /// - Catch 401 errors from backend token validation
  /// - Handle new user creation (isNewUser flag from backend)
  Future<void> signInWithGoogleProvider() async {
    _isLoadingGoogle = true;
    _errorMessage = null;
    notifyListeners();

    try {
      // Get Firebase ID token from AuthService
      final firebaseIdToken = await _authService.signInWithGoogle();
      
      print('AuthProvider: Firebase ID token received (length: ${firebaseIdToken.length})');
      
      // TODO: Send Firebase ID token to backend POST /auth/firebase-verify
      // final response = await http.post(
      //   Uri.parse('https://your-api.com/api/v1/auth/firebase-verify'),
      //   headers: {'Content-Type': 'application/json'},
      //   body: jsonEncode({'firebaseToken': firebaseIdToken}),
      // );
      // if (response.statusCode == 200) {
      //   final jwtToken = jsonDecode(response.body)['token'];
      //   _token = jwtToken;
      // } else {
      //   throw Exception('Backend verification failed: ${response.statusCode}');
      // }
      
      // For now, use Firebase ID token directly (when backend ready, replace with JWT)
      _token = firebaseIdToken;
      await _prefs.setString('token', firebaseIdToken);

      print('AuthProvider: Token saved to SharedPreferences');
      
      _isLoadingGoogle = false;
      notifyListeners();
      
      print('AuthProvider: Firebase Sign-In complete, isLoggedIn: $isLoggedIn');
    } catch (e) {
      print('AuthProvider: Firebase Sign-In failed: $e');
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