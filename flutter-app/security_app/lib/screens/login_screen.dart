import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
// import 'package:go_router/go_router.dart'; // GoRouter still needed as fallback
import 'package:provider/provider.dart'; // Import Provider
import 'package:security_app/providers/auth_provider.dart'; // Import AuthProvider
import 'package:security_app/theme/app_theme.dart';

/// Screen for user login.
class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _emailController = TextEditingController(text: 'admin');
  final _passwordController = TextEditingController(text: 'admin');

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  /// Handles login button press.
  ///
  /// Uses context.read() instead of context.watch() because we only need
  /// to call the provider, not rebuild when state changes.
  Future<void> _handleLogin() async {
    final authProvider = context.read<AuthProvider>();

    await authProvider.login(
      _emailController.text,
      _passwordController.text,
    );

    // Check mounted before using context after async operation
    if (mounted && authProvider.errorMessage != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(authProvider.errorMessage!),
          backgroundColor: Colors.red,
        ),
      );
    }

    // GoRouter redirect will automatically navigate to /home on successful login
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Login'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: _emailController,
              decoration: InputDecoration(
                labelText: 'Email',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.email),
              ),
              keyboardType: TextInputType.emailAddress,
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _passwordController,
              decoration: InputDecoration(
                labelText: 'Password',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.lock),
              ),
              obscureText: true,
            ),
            const SizedBox(height: 24),

            // Consumer rebuilds only this widget when auth state changes
            Consumer<AuthProvider>(
              builder: (context, auth, child) {
                if (auth.isLoading) {
                  return SpinKitFadingCircle(
                    color: Theme.of(context).colorScheme.primary,
                    size: 50.0,
                  );
                }

                return Container(
                  decoration: BoxDecoration(
                    gradient: kAppGradient,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      minimumSize: const Size(double.infinity, 50),
                      backgroundColor: Colors.transparent,
                      shadowColor: Colors.transparent,
                    ),
                    onPressed: _handleLogin,
                    child: const Text('Login'),
                  ),
                );
              },
            ),

            const SizedBox(height: 16),

            // Google Sign-In button with separate loading state
            Consumer<AuthProvider>(
              builder: (context, auth, child) {
                if (auth.isLoadingGoogle) {
                  return SpinKitFadingCircle(
                    color: Theme.of(context).colorScheme.primary,
                    size: 50.0,
                  );
                }

                return Container(
                  decoration: BoxDecoration(
                    gradient: kAppGradient,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      minimumSize: const Size(double.infinity, 50),
                      backgroundColor: Colors.transparent,
                      shadowColor: Colors.transparent,
                    ),
                    onPressed: () async {
                      await context.read<AuthProvider>().signInWithGoogleProvider();
                      
                      // Show error if sign-in failed
                      if (mounted && auth.errorMessage != null) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text(auth.errorMessage!),
                            backgroundColor: Colors.red,
                          ),
                        );
                      }
                    },
                    icon: auth.isLoadingGoogle 
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: SpinKitFadingCircle(color: Colors.white, size: 20.0),
                        )
                      : const FaIcon(
                          FontAwesomeIcons.google, 
                          size: 20,
                          color: Colors.white,
                        ),
                    label: auth.isLoadingGoogle 
                      ? const Text('Signing in...')
                      : RichText(
                          text: const TextSpan(
                            style: TextStyle(fontSize: 16),
                            children: [
                              TextSpan(text: 'Sign in with '),
                              TextSpan(text: 'G', style: TextStyle(color: Color(0xFF4285F4), fontWeight: FontWeight.bold)),
                              TextSpan(text: 'o', style: TextStyle(color: Color(0xFFEA4335), fontWeight: FontWeight.bold)),
                              TextSpan(text: 'o', style: TextStyle(color: Color(0xFFFBBC04), fontWeight: FontWeight.bold)),
                              TextSpan(text: 'g', style: TextStyle(color: Color(0xFF4285F4), fontWeight: FontWeight.bold)),
                              TextSpan(text: 'l', style: TextStyle(color: Color(0xFF34A853), fontWeight: FontWeight.bold)),
                              TextSpan(text: 'e', style: TextStyle(color: Color(0xFFEA4335), fontWeight: FontWeight.bold)),
                            ],
                          ),
                        ),
                  ),
                );
              },
            ),

            TextButton(
              onPressed: () {
                // TODO: Implement sign up navigation
              },
              child: const Text('Don\'t have an account? Sign up now'),
            ),
          ],
        ),
      ),
    );
  }
}