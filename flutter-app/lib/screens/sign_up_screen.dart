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
          _formKey.currentState!.validate(); // This will call _validateEmail again
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
      appBar: AppBar(
        title: const Text('Create Account'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Form(
            key: _formKey,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(height: 24),
                Text(
                  'Sign Up',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Create a new account to get started',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: Colors.grey,
                      ),
                ),
                const SizedBox(height: 32),
                TextFormField(
                  controller: _emailController,
                  decoration: InputDecoration(
                    labelText: 'Email',
                    hintText: 'example@gmail.com',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.email),
                  ),
                  keyboardType: TextInputType.emailAddress,
                  validator: _validateEmail,
                ),
                const SizedBox(height: 16),
                TextFormField(
                  controller: _displayNameController,
                  decoration: InputDecoration(
                    labelText: 'Display Name (Optional)',
                    hintText: 'Your Name',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.person),
                  ),
                  keyboardType: TextInputType.text,
                ),
                const SizedBox(height: 16),
                TextFormField(
                  controller: _passwordController,
                  decoration: InputDecoration(
                    labelText: 'Password',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.lock),
                    suffixIcon: IconButton(
                      icon: Icon(
                        _obscurePassword
                            ? Icons.visibility_off
                            : Icons.visibility,
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
                ),
                const SizedBox(height: 16),
                TextFormField(
                  controller: _confirmPasswordController,
                  decoration: InputDecoration(
                    labelText: 'Confirm Password',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.lock),
                    suffixIcon: IconButton(
                      icon: Icon(
                        _obscureConfirmPassword
                            ? Icons.visibility_off
                            : Icons.visibility,
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
                ),
                const SizedBox(height: 24),
                Consumer<AuthProvider>(
                  builder: (context, auth, child) {
                    if (auth.isLoadingSignUp) {
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
                        onPressed: _handleSignUp,
                        child: const Text('Sign Up'),
                      ),
                    );
                  },
                ),
                const SizedBox(height: 16),
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
                            style: TextStyle(fontSize: 16),
                            children: [
                              TextSpan(text: 'Sign up with '),
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
                const SizedBox(height: 24),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      'Already have an account? ',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                    TextButton(
                      onPressed: () {
                        context.go('/login');
                      },
                      child: const Text('Sign In'),
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
