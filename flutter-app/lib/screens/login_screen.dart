import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';
import 'package:security_app/providers/auth_provider.dart';
import 'package:security_app/theme/app_theme.dart';

/// Screen for user login.
class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  late TextEditingController _resetEmailController;

  @override
  void initState() {
    super.initState();
    _resetEmailController = TextEditingController();
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _resetEmailController.dispose();
    super.dispose();
  }

  /// Handles login button press.
  ///
  /// Uses context.read() instead of context.watch() because we only need
  /// to call the provider, not rebuild when state changes.
  Future<void> _handleLogin() async {
    final authProvider = context.read<AuthProvider>();

    await authProvider.loginWithEmail(
      email: _emailController.text,
      password: _passwordController.text,
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

  /// Shows a dialog to send password reset email
  void _showForgotPasswordDialog() {
    _resetEmailController.clear();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Reset Password'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
                'Enter your email address to receive a password reset link.'),
            const SizedBox(height: 16),
            TextField(
              controller: _resetEmailController,
              decoration: InputDecoration(
                labelText: 'Email',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                prefixIcon: const Icon(Icons.email),
              ),
              keyboardType: TextInputType.emailAddress,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          Consumer<AuthProvider>(
            builder: (context, auth, child) {
              if (auth.isLoading) {
                return const Padding(
                  padding: EdgeInsets.all(8.0),
                  child: SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                );
              }
              return TextButton(
                onPressed: () async {
                  final email = _resetEmailController.text.trim();
                  if (email.isEmpty) {
                    if (mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Please enter an email address'),
                          backgroundColor: Colors.red,
                        ),
                      );
                    }
                    return;
                  }

                  try {
                    await context
                        .read<AuthProvider>()
                        .sendPasswordResetEmail(email);
                    if (mounted) {
                      Navigator.pop(context);
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text(
                              'Password reset email sent! Check your inbox.'),
                          backgroundColor: Colors.green,
                        ),
                      );
                    }
                  } catch (e) {
                    if (mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text(
                              context.read<AuthProvider>().errorMessage ??
                                  'Error sending reset email'),
                          backgroundColor: Colors.red,
                        ),
                      );
                    }
                  }
                },
                child: const Text('Send'),
              );
            },
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const SizedBox(height: 60),

                // Logo/Header Section
                Center(
                  child: Container(
                    width: 80,
                    height: 80,
                    decoration: BoxDecoration(
                      color: kAccentColor,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Icon(
                      Icons.shield_outlined,
                      size: 40,
                      color: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(height: 32),

                // Title
                Text(
                  'Welcome Back',
                  style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: kTextPrimary,
                      ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 8),
                Text(
                  'Sign in to continue monitoring',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: kTextSecondary,
                      ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 48),

                // Email Field
                TextField(
                  controller: _emailController,
                  decoration: kInputDecoration(
                    labelText: 'Email',
                    hintText: 'Enter your email',
                    prefixIcon: Icons.email_outlined,
                  ),
                  keyboardType: TextInputType.emailAddress,
                  style: const TextStyle(color: kTextPrimary),
                ),
                const SizedBox(height: 16),

                // Password Field
                TextField(
                  controller: _passwordController,
                  decoration: kInputDecoration(
                    labelText: 'Password',
                    hintText: 'Enter your password',
                    prefixIcon: Icons.lock_outline,
                  ),
                  obscureText: true,
                  style: const TextStyle(color: kTextPrimary),
                ),
                const SizedBox(height: 8),

                // Forgot Password
                Align(
                  alignment: Alignment.centerRight,
                  child: TextButton(
                    onPressed: _showForgotPasswordDialog,
                    child: Text(
                      'Forgot Password?',
                      style: TextStyle(
                        color: kAccentColor,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Login Button
                Consumer<AuthProvider>(
                  builder: (context, auth, child) {
                    if (auth.isLoading) {
                      return const Center(
                        child: SpinKitFadingCircle(
                          color: kAccentColor,
                          size: 50.0,
                        ),
                      );
                    }

                    return Container(
                      height: 56,
                      decoration: BoxDecoration(
                        color: kAccentColor,
                        borderRadius: BorderRadius.circular(AppRadius.md),
                      ),
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.transparent,
                          shadowColor: Colors.transparent,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(AppRadius.md),
                          ),
                        ),
                        onPressed: _handleLogin,
                        child: const Text(
                          'Sign In',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    );
                  },
                ),

                const SizedBox(height: 24),

                // Divider
                Row(
                  children: [
                    Expanded(
                        child: Divider(color: kTextMuted.withOpacity(0.3))),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      child: Text(
                        'or continue with',
                        style: TextStyle(color: kTextMuted, fontSize: 13),
                      ),
                    ),
                    Expanded(
                        child: Divider(color: kTextMuted.withOpacity(0.3))),
                  ],
                ),

                const SizedBox(height: 24),

                // Google Sign-In Button
                Consumer<AuthProvider>(
                  builder: (context, auth, child) {
                    if (auth.isLoadingGoogle) {
                      return const Center(
                        child: SpinKitFadingCircle(
                          color: kAccentColor,
                          size: 50.0,
                        ),
                      );
                    }

                    return Container(
                      height: 56,
                      decoration: BoxDecoration(
                        color: kSurfaceColor,
                        borderRadius: BorderRadius.circular(AppRadius.md),
                        border:
                            Border.all(color: Colors.white.withOpacity(0.1)),
                      ),
                      child: ElevatedButton.icon(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.transparent,
                          shadowColor: Colors.transparent,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(AppRadius.md),
                          ),
                        ),
                        onPressed: () async {
                          await context
                              .read<AuthProvider>()
                              .signInWithGoogleProvider();

                          if (mounted && auth.errorMessage != null) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text(auth.errorMessage!),
                                backgroundColor: kErrorColor,
                              ),
                            );
                          }
                        },
                        icon: const FaIcon(
                          FontAwesomeIcons.google,
                          size: 20,
                          color: Colors.white,
                        ),
                        label: RichText(
                          text: const TextSpan(
                            style: TextStyle(
                                fontSize: 15, fontWeight: FontWeight.w500),
                            children: [
                              TextSpan(
                                  text: 'Continue with ',
                                  style: TextStyle(color: kTextSecondary)),
                              TextSpan(
                                  text: 'G',
                                  style: TextStyle(
                                      color: Color(0xFF4285F4),
                                      fontWeight: FontWeight.bold)),
                              TextSpan(
                                  text: 'o',
                                  style: TextStyle(
                                      color: Color(0xFFEA4335),
                                      fontWeight: FontWeight.bold)),
                              TextSpan(
                                  text: 'o',
                                  style: TextStyle(
                                      color: Color(0xFFFBBC04),
                                      fontWeight: FontWeight.bold)),
                              TextSpan(
                                  text: 'g',
                                  style: TextStyle(
                                      color: Color(0xFF4285F4),
                                      fontWeight: FontWeight.bold)),
                              TextSpan(
                                  text: 'l',
                                  style: TextStyle(
                                      color: Color(0xFF34A853),
                                      fontWeight: FontWeight.bold)),
                              TextSpan(
                                  text: 'e',
                                  style: TextStyle(
                                      color: Color(0xFFEA4335),
                                      fontWeight: FontWeight.bold)),
                            ],
                          ),
                        ),
                      ),
                    );
                  },
                ),

                const SizedBox(height: 32),

                // Sign Up Link
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      "Don't have an account? ",
                      style: TextStyle(color: kTextSecondary),
                    ),
                    GestureDetector(
                      onTap: () => context.go('/sign-up'),
                      child: Text(
                        'Sign Up',
                        style: TextStyle(
                          color: kAccentColor,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 32),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
