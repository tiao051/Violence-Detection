import 'package:flutter/material.dart';
import 'package:security_app/screens/tabs/camera_tab.dart';
import 'package:security_app/screens/tabs/event_tab.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  // Cannot be const because CameraTab/EventTab are stateful widgets
  static final List<Widget> _widgetOptions = <Widget>[
    CameraTab(),
    EventTab(),
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_selectedIndex == 0 ? 'Camera' : 'Events'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () {
              // TODO: Wire up logout (AuthProvider) if needed
              // Example: context.read<AuthProvider>().logout();
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
            label: 'Camera',
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