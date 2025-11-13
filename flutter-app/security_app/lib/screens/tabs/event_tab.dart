import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
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
          return Center(
            child: SpinKitFadingCircle(
              color: Theme.of(context).colorScheme.primary,
              size: 50.0,
            ),
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
                leading: SizedBox(
                  width: 100.0,
                  height: 75.0,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(8.0),
                    child: event.thumbnailUrl.isNotEmpty
                        ? Image.network(
                            event.thumbnailUrl,
                            fit: BoxFit.cover,
                            loadingBuilder: (context, child, loadingProgress) {
                              if (loadingProgress == null) return child;
                              return Container(
                                color: Colors.grey.shade800,
                                child: SpinKitFadingCircle(
                                  color: Theme.of(context).colorScheme.primary,
                                  size: 30.0,
                                ),
                              );
                            },
                            errorBuilder: (context, error, stackTrace) {
                              // Fallback to gradient placeholder on error
                              return Container(
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    colors: [
                                      const Color(0xFF0F2027),
                                      const Color(0xFF2B623A),
                                    ],
                                    begin: Alignment.topLeft,
                                    end: Alignment.bottomRight,
                                  ),
                                ),
                                child: Center(
                                  child: Icon(
                                    Icons.videocam,
                                    color: Colors.white.withOpacity(0.7),
                                    size: 32,
                                  ),
                                ),
                              );
                            },
                          )
                        : Container(
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  const Color(0xFF0F2027),
                                  const Color(0xFF2B623A),
                                ],
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                              ),
                            ),
                            child: Center(
                              child: Icon(
                                Icons.videocam,
                                color: Colors.white.withOpacity(0.7),
                                size: 32,
                              ),
                            ),
                          ),
                  ),
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