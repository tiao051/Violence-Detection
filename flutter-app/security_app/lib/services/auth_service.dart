/// Service layer for handling authentication API calls.
class AuthService {
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

  // TODO: Add register, getCameras methods here later
}