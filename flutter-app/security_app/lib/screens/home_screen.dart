import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:security_app/providers/auth_provider.dart';
import 'package:security_app/providers/camera_provider.dart';
import 'package:security_app/providers/event_provider.dart';
import 'package:security_app/screens/tabs/camera_tab.dart';
import 'package:security_app/screens/tabs/event_tab.dart';
import 'package:security_app/services/notification_service.dart';

/// Home screen that exposes the app's primary tabs (Cameras, Events).
///
/// The screen keeps the authentication logout action in the app bar and
/// initializes lightweight services that should run once per session (e.g.
/// notifications). Widgets used in the tab list are stateful, so the list is
/// not declared `const`.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  // Tabs are stateful widgets; therefore the list cannot be const.
  static final List<Widget> _widgetOptions = <Widget>[
    CameraTab(),
    EventTab(),
  ];

  // Lightweight service to handle FCM initialization.
  // We keep the service as a field so it can be reused or extended later.
  final NotificationService _notificationService = NotificationService();

  @override
  void initState() {
    super.initState();

    // Initialize notifications once when HomeScreen is created. We intentionally
    // do not await here to avoid delaying the UI; initialization is non-blocking.
    // Any errors are logged in debug mode by the service itself.
    _notificationService.initialize();
  }

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_selectedIndex == 0 ? 'Cameras' : 'Events'),
        flexibleSpace: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                const Color(0xFF0F2027), // Dark background
                const Color(0xFF2B623A), // Green accent
              ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
        ),
        actions: [
          // Logout action uses AuthProvider to clear session; routing is
          // handled by GoRouter via the global auth listener.
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () {
              // Clear all cached data to prevent stale data leaks between users
              context.read<CameraProvider>().clearCache();
              context.read<EventProvider>().clearCache();
              context.read<AuthProvider>().logout();
            },
          ),
        ],
      ),
      body: Center(
        child: _widgetOptions.elementAt(_selectedIndex),
      ),
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.videocam),
            label: 'Cameras',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.warning),
            label: 'Events',
          ),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: Colors.deepPurple,
        onTap: _onItemTapped,
      ),
    );
  }
}