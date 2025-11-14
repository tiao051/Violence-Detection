import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';

/// Service layer for handling authentication API calls.
class AuthService {
  // Firebase Auth instance
  final FirebaseAuth _firebaseAuth = FirebaseAuth.instance;

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

  /// Initiates Google Sign-In via Firebase Authentication.
  ///
  /// Flow:
  /// 1. GoogleSignIn SDK shows Google account picker
  /// 2. User selects account
  /// 3. Firebase receives Google credential
  /// 4. Firebase verifies with Google and creates/updates user
  /// 5. Returns Firebase ID token (JWT from Firebase)
  /// 6. Token is verified by backend at POST /auth/firebase-verify
  ///
  /// Returns the Firebase ID token to be sent to backend for verification.
  /// Throws [Exception] if sign-in fails or user cancels.
  Future<String> signInWithGoogle() async {
    try {
      print('AuthService: Starting Google Sign-In via Firebase...');

      // Step 1: Use GoogleSignIn to get Google credential
      final GoogleSignIn googleSignIn = GoogleSignIn();
      final GoogleSignInAccount? googleUser = await googleSignIn.signIn();

      if (googleUser == null) {
        throw Exception('Google Sign-In cancelled by user');
      }

      print('AuthService: Google account selected: ${googleUser.email}');

      // Step 2: Get Google authentication tokens
      final GoogleSignInAuthentication googleAuth = await googleUser.authentication;

      // Step 3: Create Firebase credential from Google tokens
      final AuthCredential credential = GoogleAuthProvider.credential(
        accessToken: googleAuth.accessToken,
        idToken: googleAuth.idToken,
      );

      // Step 4: Sign in to Firebase with Google credential
      final UserCredential userCredential =
          await _firebaseAuth.signInWithCredential(credential);

      final User? user = userCredential.user;

      if (user == null) {
        throw Exception('Firebase: User is null after sign-in');
      }

      print('AuthService: Firebase Sign-In success: ${user.email}');
      print('AuthService: User UID: ${user.uid}');
      print('AuthService: New user: ${userCredential.additionalUserInfo?.isNewUser}');

      // Step 5: Get Firebase ID token (JWT)
      // This token contains user claims and can be verified by backend
      final String? idToken = await user.getIdToken();

      if (idToken == null) {
        throw Exception('Failed to get Firebase ID token');
      }

      print('AuthService: Firebase ID Token retrieved (length: ${idToken.length})');

      return idToken;
    } on FirebaseAuthException catch (e) {
      print('AuthService: Firebase Auth Error - Code: ${e.code}, Message: ${e.message}');
      throw Exception('Firebase Sign-In failed: ${e.message}');
    } catch (e) {
      print('AuthService: Unknown Error: $e');
      throw Exception('Google Sign-In failed: $e');
    }
  }

  /// Signs out from Google account and Firebase.
  ///
  /// Called during logout() to sign out from both Google and Firebase,
  /// ensuring user must pick account again on next sign-in.
  Future<void> signOutGoogle() async {
    try {
      // Sign out from Google Sign-In SDK (disconnects cached account)
      final GoogleSignIn googleSignIn = GoogleSignIn();
      await googleSignIn.disconnect();
      print('AuthService: Disconnected from Google Sign-In');

      // Sign out from Firebase
      await _firebaseAuth.signOut();
      print('AuthService: Signed out from Firebase');
    } catch (e) {
      print('AuthService: Sign out error: $e');
    }
  }

  // TODO: Add register, getCameras methods here later
}