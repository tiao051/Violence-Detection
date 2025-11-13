import 'package:google_sign_in/google_sign_in.dart';

/// Service layer for handling authentication API calls.
class AuthService {
  // Create single instance of GoogleSignIn (reusable across app)
  final GoogleSignIn _googleSignIn = GoogleSignIn();

  /// Simulates login API call with hardcoded credentials for development.
  ///
  /// Uses a 2-second delay to mimic real network latency.
  /// Accepts only 'admin'/'admin' credentials until backend is integrated.
  ///
  /// Returns a mock JWT token on success.
  /// Throws [Exception] on invalid credentials.
  Future<String> login(String email, String password) async {
    await Future.delayed(const Duration(seconds: 2));

    if (email == 'admin' && password == 'admin') {
      print("AuthService: Login successful, returning fake token.");
      return 'fake_jwt_token_1234567890';
    } else {
      print("AuthService: Invalid credentials.");
      throw Exception('Invalid email or password');
    }
  }

  /// Initiates Google Sign-In flow and returns mock token.
  ///
  /// Flow:
  /// 1. Calls GoogleSignIn.signIn() to show Google account picker
  /// 2. If user cancels (googleUser == null), returns false
  /// 3. If successful, prints email to console as proof
  /// 4. Returns fake token (mock) since backend doesn't exist yet
  ///
  /// Throws [Exception] if sign-in fails.
  Future<String> signInWithGoogle() async {
    try {
      // Show Google account picker pop-up
      final GoogleSignInAccount? googleUser = await _googleSignIn.signIn();

      // User cancelled the sign-in (hit back button)
      if (googleUser == null) {
        throw Exception('Google Sign-In cancelled by user');
      }

      // Sign-in successful! Print email as proof
      print('Google Sign-In Success: ${googleUser.email}');

      // MOCK: Return fake token (backend doesn't exist yet)
      // In production: send googleUser.authentication.idToken to POST /auth/google
      return 'fake_google_token_123';
    } catch (e) {
      print('Google Sign-In Error: $e');
      throw Exception('Google Sign-In failed: $e');
    }
  }

  /// Signs out from Google account.
  ///
  /// Called during logout() to "forget" the Google account,
  /// ensuring user must pick account again on next sign-in.
  Future<void> signOutGoogle() async {
    await _googleSignIn.signOut();
  }

  // TODO: Add register, getCameras methods here later
}