import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:security_app/models/auth_model.dart';

/// Service layer for handling authentication API calls.
class AuthService {
  // Firebase Auth instance
  final FirebaseAuth _firebaseAuth = FirebaseAuth.instance;

  // Firestore instance for storing user data
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

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

  /// Signs up a new user with email and password
  ///
  /// Flow:
  /// 1. Create user with email/password in Firebase Auth
  /// 2. Update display name in Firebase Auth
  /// 3. Save user data to Firestore (users/{uid})
  ///
  /// Returns the Firebase user's UID on success
  /// Throws [Exception] if email already exists or validation fails
  Future<String> signUpWithEmail({
    required String email,
    required String password,
    required String displayName,
  }) async {
    try {
      print('AuthService: Starting email sign up for: $email');

      // Step 1: Create user with email and password
      final UserCredential userCredential =
          await _firebaseAuth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );

      final User? user = userCredential.user;

      if (user == null) {
        throw Exception('Firebase: User is null after sign up');
      }

      print('AuthService: User created: ${user.uid}');

      // Step 2: Update display name
      await user.updateDisplayName(displayName);
      print('AuthService: Display name updated: $displayName');

      // Step 3: Save user data to Firestore
      final authModel = AuthModel(
        uid: user.uid,
        email: email,
        displayName: displayName,
        photoUrl: null,
        authProvider: 'email',
        createdAt: DateTime.now(),
      );

      await _firestore.collection('users').doc(user.uid).set(authModel.toJson());
      print('AuthService: User data saved to Firestore');

      return user.uid;
    } on FirebaseAuthException catch (e) {
      print('AuthService: Firebase Auth Error - Code: ${e.code}, Message: ${e.message}');
      if (e.code == 'email-already-in-use') {
        throw Exception('Email already registered');
      } else if (e.code == 'weak-password') {
        throw Exception('Password is too weak');
      }
      throw Exception('Sign up failed: ${e.message}');
    } catch (e) {
      print('AuthService: Sign up error: $e');
      throw Exception('Sign up failed: $e');
    }
  }

  /// Logs in user with email and password
  ///
  /// Flow:
  /// 1. Sign in to Firebase Auth with email/password
  /// 2. Get user from Firebase
  ///
  /// Returns the Firebase user's UID on success
  /// Throws [Exception] if credentials are invalid
  Future<String> loginWithEmail({
    required String email,
    required String password,
  }) async {
    try {
      print('AuthService: Starting email login for: $email');

      // Sign in to Firebase Auth
      final UserCredential userCredential =
          await _firebaseAuth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      final User? user = userCredential.user;

      if (user == null) {
        throw Exception('Firebase: User is null after login');
      }

      print('AuthService: Login successful: ${user.uid}');
      return user.uid;
    } on FirebaseAuthException catch (e) {
      print('AuthService: Firebase Auth Error - Code: ${e.code}, Message: ${e.message}');
      if (e.code == 'user-not-found') {
        throw Exception('Email not registered');
      } else if (e.code == 'wrong-password') {
        throw Exception('Incorrect password');
      }
      throw Exception('Login failed: ${e.message}');
    } catch (e) {
      print('AuthService: Login error: $e');
      throw Exception('Login failed: $e');
    }
  }

  /// Changes password for the current user
  ///
  /// Flow:
  /// 1. Get current user
  /// 2. Re-authenticate with current password
  /// 3. Update password
  ///
  /// Throws [Exception] if current password is wrong or update fails
  Future<void> changePassword({
    required String currentPassword,
    required String newPassword,
  }) async {
    try {
      print('AuthService: Starting password change');

      final User? user = _firebaseAuth.currentUser;

      if (user == null || user.email == null) {
        throw Exception('No user is currently logged in');
      }

      // Re-authenticate user with current password
      final AuthCredential credential = EmailAuthProvider.credential(
        email: user.email!,
        password: currentPassword,
      );

      try {
        await user.reauthenticateWithCredential(credential);
        print('AuthService: Re-authentication successful');
      } on FirebaseAuthException catch (e) {
        if (e.code == 'wrong-password') {
          throw Exception('Current password is incorrect');
        }
        throw Exception('Re-authentication failed: ${e.message}');
      }

      // Update password
      await user.updatePassword(newPassword);
      print('AuthService: Password updated successfully');
    } catch (e) {
      print('AuthService: Change password error: $e');
      rethrow;
    }
  }

  /// Updates the display name of the current user
  ///
  /// Flow:
  /// 1. Get current user
  /// 2. Update display name in Firebase Auth
  /// 3. Update display name in Firestore
  ///
  /// Throws [Exception] if no user is logged in
  Future<void> updateDisplayName(String newDisplayName) async {
    try {
      print('AuthService: Updating display name to: $newDisplayName');

      final User? user = _firebaseAuth.currentUser;

      if (user == null) {
        throw Exception('No user is currently logged in');
      }

      // Update Firebase Auth profile
      await user.updateDisplayName(newDisplayName);
      print('AuthService: Firebase Auth display name updated');

      // Update Firestore
      await _firestore.collection('users').doc(user.uid).update({
        'displayName': newDisplayName,
        'updatedAt': DateTime.now().toIso8601String(),
      });
      print('AuthService: Firestore display name updated');
    } catch (e) {
      print('AuthService: Update display name error: $e');
      rethrow;
    }
  }

  /// Fetches the current user's profile from Firestore
  ///
  /// Returns AuthModel with user data, or null if user not found
  /// Throws [Exception] if no user is logged in
  Future<AuthModel?> getUserProfile() async {
    try {
      print('AuthService: Fetching user profile');

      final User? user = _firebaseAuth.currentUser;

      if (user == null) {
        throw Exception('No user is currently logged in');
      }

      final DocumentSnapshot userDoc =
          await _firestore.collection('users').doc(user.uid).get();

      if (!userDoc.exists) {
        print('AuthService: User profile not found in Firestore');
        return null;
      }

      final authModel = AuthModel.fromJson(userDoc.data() as Map<String, dynamic>);
      print('AuthService: User profile fetched successfully');
      return authModel;
    } catch (e) {
      print('AuthService: Get user profile error: $e');
      rethrow;
    }
  }

  /// Checks if an email is already registered
  ///
  /// Returns true if email exists, false otherwise
  /// Uses Firebase Auth API to fetch sign-in methods
  Future<bool> emailExists(String email) async {
    try {
      print('AuthService: Checking if email exists: $email');

      final signInMethods = await _firebaseAuth.fetchSignInMethodsForEmail(email);
      final exists = signInMethods.isNotEmpty;

      print('AuthService: Email exists: $exists');
      return exists;
    } catch (e) {
      print('AuthService: Error checking email: $e');
      return false;
    }
  }

  /// Gets the Firebase ID token for the current user
  ///
  /// Returns the ID token string, or null if no user is logged in
  Future<String?> getIdToken() async {
    try {
      final User? user = _firebaseAuth.currentUser;
      if (user == null) {
        return null;
      }

      final String? idToken = await user.getIdToken();
      print('AuthService: ID token retrieved');
      return idToken;
    } catch (e) {
      print('AuthService: Error getting ID token: $e');
      return null;
    }
  }

  // TODO: Add getCameras methods here later
}