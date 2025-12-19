import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
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
  int _rebuildKey = 0; // Force rebuild counter

  // Tabs are stateful widgets; therefore the list cannot be const.
  List<Widget> get _widgetOptions => [
        const CameraTab(),
        EventTab(
            key: ValueKey(
                'event_tab_$_rebuildKey')), // Dynamic key forces rebuild
      ];

  // Lightweight service to handle FCM initialization.
  // We keep the service as a field so it can be reused or extended later.
  final NotificationService _notificationService = NotificationService();

  @override
  void initState() {
    super.initState();

    // Initialize notifications with deep link handler for event details
    _notificationService.initialize(
      onNotificationTap: (eventId) async {
        // Navigate to event detail when notification is tapped
        // Use context.push() to add to navigation stack
        if (mounted) {
          // Find the event in provider and navigate to it
          final eventProvider = context.read<EventProvider>();
          try {
            final event = eventProvider.events.firstWhere(
              (e) => e.id == eventId,
            );
            await context.push('/event_detail', extra: event);
            // Force EventTab rebuild when back
            setState(() {
              _rebuildKey++;
            });
          } catch (e) {
            // Event not found in list, could be loading or not yet fetched
            if (kDebugMode) {
              print('Event $eventId not found in provider');
            }
          }
        }
      },
    );
  }

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  /// Debug method to test notification deep linking.
  ///
  /// Shows a dialog with unviewed events that can be tapped to simulate
  /// notification tap behavior. Only available in debug mode.
  void _showTestNotificationDialog() {
    final eventProvider = context.read<EventProvider>();
    final unviewedEvents = eventProvider.unviewedEvents;

    if (unviewedEvents.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('No unviewed events to test with'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    showDialog(
      context: context,
      builder: (context) => SimpleDialog(
        title: const Text('🧪 Test Notification'),
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Text(
              'Tap event to simulate notification:',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
          ),
          const SizedBox(height: 12),
          ...unviewedEvents.map((event) => SimpleDialogOption(
                onPressed: () async {
                  Navigator.pop(context);
                  // Navigate and wait for return
                  await context.push('/event_detail', extra: event);
                  // Force EventTab rebuild when back
                  setState(() {
                    _rebuildKey++;
                  });
                },
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      event.cameraName,
                      style: const TextStyle(fontWeight: FontWeight.w500),
                    ),
                    Text(
                      'ID: ${event.id}',
                      style:
                          TextStyle(fontSize: 12, color: Colors.grey.shade600),
                    ),
                  ],
                ),
              )),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_selectedIndex == 0 ? 'Cameras' : 'Events'),
        actions: [
          // Debug button to test deep linking (only in debug mode)
          if (kDebugMode && _selectedIndex == 1) // Only show on Events tab
            Tooltip(
              message: 'Test Notification Deep Link',
              child: IconButton(
                icon: const Icon(Icons.developer_mode),
                onPressed: () {
                  if (kDebugMode) print('DEBUG: Test button pressed');
                  _showTestNotificationDialog();
                },
              ),
            ),
          // Settings button
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () {
              context.push('/settings');
            },
          ),
          // Profile button
          IconButton(
            icon: const Icon(Icons.person_outline),
            onPressed: () {
              context.push('/profile');
            },
          ),
        ],
      ),
      body: _widgetOptions.elementAt(_selectedIndex),
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.videocam_outlined),
            activeIcon: Icon(Icons.videocam),
            label: 'Cameras',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.notification_important_outlined),
            activeIcon: Icon(Icons.notification_important),
            label: 'Events',
          ),
        ],
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        // Uses theme's bottomNavigationBarTheme (kAccentColor)
      ),
    );
  }
}
