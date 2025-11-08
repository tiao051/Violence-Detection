import 'package:flutter/material.dart';
// import 'package:go_router/go_router.dart'; // GoRouter still needed as fallback
import 'package:provider/provider.dart'; // Import Provider
import 'package:security_app/providers/auth_provider.dart'; // Import AuthProvider

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
                  return const CircularProgressIndicator();
                }

                return ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    minimumSize: const Size(double.infinity, 50),
                  ),
                  onPressed: _handleLogin,
                  child: const Text('Login'),
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