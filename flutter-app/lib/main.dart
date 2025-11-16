import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart'; 
import 'package:security_app/providers/auth_provider.dart';
import 'package:security_app/providers/camera_provider.dart';
import 'package:security_app/providers/event_provider.dart';
import 'package:security_app/providers/settings_provider.dart';
import 'package:security_app/providers/profile_provider.dart';
import 'package:security_app/screens/event_detail_screen.dart';
import 'package:security_app/screens/home_screen.dart';
import 'package:security_app/screens/live_view_screen.dart';
import 'package:security_app/screens/login_screen.dart';
import 'package:security_app/screens/sign_up_screen.dart';
import 'package:security_app/screens/settings_screen.dart';
import 'package:security_app/screens/profile_screen.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:security_app/theme/app_theme.dart'; 

void main() async {
  // Initialize Flutter bindings before async operations in main()
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize Firebase before using any Firebase services
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  
  final prefs = await SharedPreferences.getInstance();
  runApp(MyApp(prefs: prefs));
}

class MyApp extends StatelessWidget {
  final SharedPreferences prefs;
  const MyApp({super.key, required this.prefs});

  @override
  Widget build(BuildContext context) {
    // Using MultiProvider to make all state providers available app-wide
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(
          create: (context) => AuthProvider(prefs),
        ),
        ChangeNotifierProvider(
          create: (context) => CameraProvider(),
        ),
        ChangeNotifierProvider(
          create: (context) => EventProvider(),
        ),
        ChangeNotifierProvider(
          create: (context) => SettingsProvider(),
        ),
        ChangeNotifierProvider(
          create: (context) => ProfileProvider(),
        ),
      ],
      // Consumer rebuilds MaterialApp when auth state changes to update routes
      child: Consumer<AuthProvider>(
        builder: (context, authProvider, child) {
          return MaterialApp.router(
            title: 'Security App',
            routerConfig: _buildRouter(authProvider),
            theme: appThemeData,
          );
        },
      ),
    );
  }
}

/// Builds the application's GoRouter using the provided [AuthProvider].
///
/// The router listens to [authProvider] to perform redirects based on
/// authentication state (e.g. redirecting to /login when not authenticated).
GoRouter _buildRouter(AuthProvider authProvider) {
  return GoRouter(
    initialLocation: '/login',
    refreshListenable: authProvider, 
    routes: [
      GoRoute(
        path: '/login',
        builder: (context, state) => const LoginScreen(),
      ),
      GoRoute(
        path: '/sign-up',
        builder: (context, state) => const SignUpScreen(),
      ),
      GoRoute(
        path: '/home',
        builder: (context, state) => const HomeScreen(),
      ),
      GoRoute(
        path: '/settings',
        builder: (context, state) => const SettingsScreen(),
      ),
      GoRoute(
        path: '/profile',
        builder: (context, state) => const ProfileScreen(),
      ),
      GoRoute(
        path: '/live_view/:cameraId',
        builder: (context, state) {
          final cameraId = state.pathParameters['cameraId'] ?? 'INVALID_ID';
          return LiveViewScreen(cameraId: cameraId);
        },
      ),
      // Event detail route uses state.extra to pass event object
      // because GoRouter doesn't serialize complex objects in path parameters
      GoRoute(
        path: '/event_detail',
        builder: (context, state) {
          final event = state.extra as dynamic;

          if (event != null) {
            return EventDetailScreen(event: event);
          } else {
            return const Scaffold(
              body: Center(child: Text('Error: No event selected')),
            );
          }
        },
      ),
    ],
    redirect: (BuildContext context, GoRouterState state) {
      
      // FIX 1: If sign up is in progress, DON'T redirect.
      if (authProvider.isLoadingSignUp) {
        return null;
      }
      
      // FIX 2: If changing password, DON'T redirect.
      // This prevents logout when re-authentication fails.
      if (authProvider.isChangingPassword) {
        return null;
      }
      
      // FIX 3: If there is an error message, DON'T redirect.
      // Stay on the page to let the UI show the error.
      if (authProvider.errorMessage != null) {
        return null;
      }

      // --- Original Logic (using matchedLocation) ---
      final isLoggedIn = authProvider.isLoggedIn;
      
      // Use matchedLocation, NOT state.uri.toString()
      final loggingIn = state.matchedLocation == '/login';
      final signingUp = state.matchedLocation == '/sign-up';

      // Debug print
      print('GoRouter Redirect: location=${state.matchedLocation}, isLoggedIn=$isLoggedIn, signingUp=$signingUp');

      // Prevent unauthenticated users from accessing protected routes
      if (!isLoggedIn && !loggingIn && !signingUp) {
        print('GoRouter Redirect: *** ĐANG CHUYỂN HƯỚNG VỀ /LOGIN ***');
        return '/login';
      }

      // Prevent authenticated users from seeing login or sign-up screens
      if (isLoggedIn && (loggingIn || signingUp)) {
        return '/home';
      }

      return null; // Allow navigation
    },
  );
}