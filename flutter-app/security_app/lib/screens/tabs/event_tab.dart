import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import 'package:security_app/providers/event_provider.dart';

/// Tab that displays detected events (alarms) from cameras.
class EventTab extends StatefulWidget {
  const EventTab({super.key});

  @override
  State<EventTab> createState() => _EventTabState();
}

class _EventTabState extends State<EventTab> {

  @override
  void initState() {
    super.initState();
    // Defer fetch until after first frame to avoid calling provider during build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<EventProvider>().fetchEvents();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<EventProvider>(
      builder: (context, eventProvider, child) {
        if (eventProvider.isLoading) {
          return const Center(
            child: CircularProgressIndicator(),
          );
        }

        if (eventProvider.errorMessage != null) {
          return Center(
            child: Text('Error: ${eventProvider.errorMessage}'),
          );
        }

        final events = eventProvider.events;

        if (events.isEmpty) {
          return const Center(child: Text('No events found.'));
        }

        return ListView.builder(
          itemCount: events.length,
          itemBuilder: (context, index) {
            final event = events[index];
            final formattedTime = DateFormat('HH:mm - dd/MM/yyyy')
                                  .format(event.timestamp);

            return Card(
              margin: const EdgeInsets.all(8.0),
              child: ListTile(
                leading: CircleAvatar(
                  backgroundColor: Colors.red.shade100,
                  child: const Icon(Icons.warning_amber, color: Colors.red),
                ),
                title: Text('Detected at ${event.cameraName}'),
                subtitle: Text(formattedTime),
                trailing: const Icon(Icons.chevron_right),
                onTap: () {
                  // Use extra to pass event object since GoRouter can't serialize complex objects in path
                  context.push('/event_detail', extra: event);
                },
              ),
            );
          },
        );
      },
    );
  }
}