import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';
import 'package:security_app/providers/auth_provider.dart';
import 'package:security_app/theme/app_theme.dart';

/// Screen for user registration with email and password.
class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final _formKey = GlobalKey<FormState>();
  late TextEditingController _emailController;
  late TextEditingController _passwordController;
  late TextEditingController _confirmPasswordController;
  late TextEditingController _displayNameController;

  bool _obscurePassword = true;
  bool _obscureConfirmPassword = true;
  String? _serverEmailError; // To hold server-side email errors

  @override
  void initState() {
    super.initState();
    _emailController = TextEditingController();
    _passwordController = TextEditingController();
    _confirmPasswordController = TextEditingController();
    _displayNameController = TextEditingController();
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    _displayNameController.dispose();
    super.dispose();
  }

  /// Email validator
  String? _validateEmail(String? value) {
    // Priority 1: Check for server-side errors
    // If we have a server error, show it and consume it
    if (_serverEmailError != null) {
      return _serverEmailError;
    }

    // Priority 2: Standard client-side validation
    if (value == null || value.isEmpty) {
      return 'Email is required';
    }
    if (!RegExp(r'^[^@]+@[^@]+\.[^@]+').hasMatch(value)) {
      return 'Please enter a valid email';
    }
    return null;
  }

  /// Password validator
  String? _validatePassword(String? value) {
    if (value == null || value.isEmpty) {
      return 'Password is required';
    }
    if (value.length < 6) {
      return 'Password must be at least 6 characters';
    }
    return null;
  }

  /// Confirm password validator
  String? _validateConfirmPassword(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please confirm your password';
    }
    if (value != _passwordController.text) {
      return 'Passwords do not match';
    }
    return null;
  }

  /// Handle sign up button press
  Future<void> _handleSignUp() async {
    print('SignUpScreen: Sign up button pressed');

    if (_serverEmailError != null) {
      setState(() {
        _serverEmailError = null;
      });
    }

    // 1. Run client-side validation
    if (!_formKey.currentState!.validate()) {
      print('SignUpScreen: Form validation failed');
      return;
    }

    print('SignUpScreen: Form validation passed, calling provider...');

    final authProvider = context.read<AuthProvider>();
    final email = _emailController.text.trim();
    final displayName = _displayNameController.text.trim().isEmpty
        ? 'User'
        : _displayNameController.text.trim();

    print('SignUpScreen: Email: $email, DisplayName: $displayName');

    // 2. Call the provider (NO try-catch here)
    await authProvider.signUp(
      email: email,
      password: _passwordController.text,
      displayName: displayName,
    );

    // 3. Handle the response
    if (mounted) {
      final errorMessage = authProvider.errorMessage;

      if (errorMessage != null) {
        // ERROR CASE
        print('SignUpScreen: Signup error from provider: $errorMessage');

        if (errorMessage.contains('Email already registered')) {
          // A: THIS IS THE KEY
          // Store the error...
          setState(() {
            _serverEmailError = errorMessage;
          });
          // ...and force the form to re-validate
          _formKey.currentState!
              .validate(); // This will call _validateEmail again
        } else {
          // B: For all other errors (network, weak-password, etc.)
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(errorMessage),
              backgroundColor: Colors.red,
              duration: const Duration(seconds: 3),
            ),
          );
        }
      } else if (authProvider.isLoggedIn) {
        // SUCCESS CASE
        print('SignUpScreen: Signup successful, redirecting to /home');
        context.go('/home');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24.0),
            child: Form(
              key: _formKey,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 40),

                  // Logo/Header Section
                  Center(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(20),
                      child: Image.asset(
                        'assets/images/app_logo.png',
                        width: 100,
                        height: 100,
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),

                  // Title
                  Text(
                    'Create Account',
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: kTextPrimary,
                        ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Sign up to start monitoring',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: kTextSecondary,
                        ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 32),

                  // Email Field
                  TextFormField(
                    controller: _emailController,
                    decoration: kInputDecoration(
                      labelText: 'Email',
                      hintText: 'example@gmail.com',
                      prefixIcon: Icons.email_outlined,
                    ),
                    keyboardType: TextInputType.emailAddress,
                    validator: _validateEmail,
                    style: const TextStyle(color: kTextPrimary),
                  ),
                  const SizedBox(height: 16),

                  // Display Name Field
                  TextFormField(
                    controller: _displayNameController,
                    decoration: kInputDecoration(
                      labelText: 'Display Name (Optional)',
                      hintText: 'Your Name',
                      prefixIcon: Icons.person_outline,
                    ),
                    keyboardType: TextInputType.text,
                    style: const TextStyle(color: kTextPrimary),
                  ),
                  const SizedBox(height: 16),

                  // Password Field
                  TextFormField(
                    controller: _passwordController,
                    decoration: kInputDecoration(
                      labelText: 'Password',
                      hintText: 'At least 6 characters',
                      prefixIcon: Icons.lock_outline,
                      suffixIcon: IconButton(
                        icon: Icon(
                          _obscurePassword
                              ? Icons.visibility_off_outlined
                              : Icons.visibility_outlined,
                          color: kTextSecondary,
                        ),
                        onPressed: () {
                          setState(() {
                            _obscurePassword = !_obscurePassword;
                          });
                        },
                      ),
                    ),
                    obscureText: _obscurePassword,
                    validator: _validatePassword,
                    style: const TextStyle(color: kTextPrimary),
                  ),
                  const SizedBox(height: 16),

                  // Confirm Password Field
                  TextFormField(
                    controller: _confirmPasswordController,
                    decoration: kInputDecoration(
                      labelText: 'Confirm Password',
                      hintText: 'Re-enter your password',
                      prefixIcon: Icons.lock_outline,
                      suffixIcon: IconButton(
                        icon: Icon(
                          _obscureConfirmPassword
                              ? Icons.visibility_off_outlined
                              : Icons.visibility_outlined,
                          color: kTextSecondary,
                        ),
                        onPressed: () {
                          setState(() {
                            _obscureConfirmPassword = !_obscureConfirmPassword;
                          });
                        },
                      ),
                    ),
                    obscureText: _obscureConfirmPassword,
                    validator: _validateConfirmPassword,
                    style: const TextStyle(color: kTextPrimary),
                  ),
                  const SizedBox(height: 32),

                  // Sign Up Button
                  Consumer<AuthProvider>(
                    builder: (context, auth, child) {
                      if (auth.isLoadingSignUp) {
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
                          onPressed: _handleSignUp,
                          child: const Text(
                            'Create Account',
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

                  // Google Sign Up Button
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
                            if (mounted &&
                                context.read<AuthProvider>().isLoggedIn) {
                              context.go('/home');
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

                  // Sign In Link
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'Already have an account? ',
                        style: TextStyle(color: kTextSecondary),
                      ),
                      GestureDetector(
                        onTap: () => context.go('/login'),
                        child: Text(
                          'Sign In',
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
      ),
    );
  }
}
