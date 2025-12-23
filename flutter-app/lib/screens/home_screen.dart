import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';
import 'package:security_app/providers/event_provider.dart';
import 'package:security_app/providers/settings_provider.dart';
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
            debugPrint('Event $eventId not found in provider');
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_selectedIndex == 0 ? 'Cameras' : 'Events'),
        actions: [
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
      bottomNavigationBar: Consumer2<EventProvider, SettingsProvider>(
        builder: (context, eventProvider, settingsProvider, child) {
          final unviewedCount = eventProvider.unviewedCount;
          final showBadge = settingsProvider.showBadge && unviewedCount > 0;

          return Stack(
            alignment: Alignment.center,
            children: [
              BottomNavigationBar(
                items: <BottomNavigationBarItem>[
                  const BottomNavigationBarItem(
                    icon: Icon(Icons.videocam_outlined),
                    activeIcon: Icon(Icons.videocam),
                    label: 'Cameras',
                  ),
                  BottomNavigationBarItem(
                    icon: Badge(
                      isLabelVisible: showBadge,
                      label: Text(
                        unviewedCount > 99 ? '99+' : unviewedCount.toString(),
                        style: const TextStyle(
                            fontSize: 10, fontWeight: FontWeight.bold),
                      ),
                      child: const Icon(Icons.notification_important_outlined),
                    ),
                    activeIcon: Badge(
                      isLabelVisible: showBadge,
                      label: Text(
                        unviewedCount > 99 ? '99+' : unviewedCount.toString(),
                        style: const TextStyle(
                            fontSize: 10, fontWeight: FontWeight.bold),
                      ),
                      child: const Icon(Icons.notification_important),
                    ),
                    label: 'Events',
                  ),
                ],
                currentIndex: _selectedIndex,
                onTap: _onItemTapped,
              ),
              // Vertical divider in center
              Positioned(
                top: 12,
                bottom: 12,
                child: Container(
                  width: 1,
                  color: Colors.white.withOpacity(0.1),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
