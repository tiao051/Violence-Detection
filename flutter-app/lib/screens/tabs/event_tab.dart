import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../../providers/event_provider.dart';
import '../../theme/app_theme.dart';
import '../../widgets/error_widget.dart' as error_widget;
import '../../widgets/empty_state_widget.dart';

/// Tab that displays detected events (alarms) from cameras.
class EventTab extends StatefulWidget {
  const EventTab({super.key});

  @override
  State<EventTab> createState() => _EventTabState();
}

class _EventTabState extends State<EventTab> with WidgetsBindingObserver {

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Defer fetch until after first frame to avoid calling provider during build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<EventProvider>().fetchEvents();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  /// Called when app lifecycle changes (resume, pause, etc)
  /// Force rebuild when app resumes to get latest viewed state
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      print('🔄 EventTab: App resumed - rebuilding to get latest state');
      setState(() {});
    }
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<EventProvider>(
      builder: (context, eventProvider, child) {
        print('🎨 EventTab rebuild - unviewed: ${eventProvider.unviewedCount}');
        
        if (eventProvider.isLoading) {
          return Center(
            child: SpinKitFadingCircle(
              color: Theme.of(context).colorScheme.primary,
              size: 50.0,
            ),
          );
        }

        if (eventProvider.errorMessage != null) {
          return error_widget.ErrorWidget(
            errorMessage: eventProvider.errorMessage ?? "Unknown error",
            onRetry: () {
              eventProvider.clearCache();
              eventProvider.fetchEvents();
            },
            iconData: Icons.warning_rounded,
            title: "Failed to Load Events",
          );
        }

        final events = eventProvider.events;

        if (events.isEmpty) {
          return EmptyStateWidget(
            title: "No Events Yet",
            subtitle: "Events detected by cameras will appear here",
            iconData: Icons.event_outlined,
            showRefreshHint: true,
          );
        }

        final filteredEvents = eventProvider.filteredEvents;

        return Column(
          children: [
            // Unviewed events badge
            if (eventProvider.unviewedCount > 0)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
                color: Theme.of(context).colorScheme.error.withOpacity(0.1),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12.0,
                        vertical: 6.0,
                      ),
                      decoration: BoxDecoration(
                        color: Theme.of(context).colorScheme.error,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        '${eventProvider.unviewedCount} unviewed',
                        style: Theme.of(context).textTheme.labelSmall?.copyWith(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            // Date filter chips
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Row(
                  children: [
                    FilterChip(
                      label: const Text('Today'),
                      selected: eventProvider.dateFilter == DateFilter.today,
                      onSelected: (_) {
                        eventProvider.setDateFilter(DateFilter.today);
                      },
                    ),
                    const SizedBox(width: 8),
                    FilterChip(
                      label: const Text('This Week'),
                      selected:
                          eventProvider.dateFilter == DateFilter.thisWeek,
                      onSelected: (_) {
                        eventProvider.setDateFilter(DateFilter.thisWeek);
                      },
                    ),
                    const SizedBox(width: 8),
                    FilterChip(
                      label: const Text('This Month'),
                      selected:
                          eventProvider.dateFilter == DateFilter.thisMonth,
                      onSelected: (_) {
                        eventProvider.setDateFilter(DateFilter.thisMonth);
                      },
                    ),
                    const SizedBox(width: 8),
                    FilterChip(
                      label: const Text('All'),
                      selected: eventProvider.dateFilter == DateFilter.all,
                      onSelected: (_) {
                        eventProvider.setDateFilter(DateFilter.all);
                      },
                    ),
                  ],
                ),
              ),
            ),
            // Event list
            Expanded(
              child: RefreshIndicator(
                onRefresh: () => eventProvider.refreshEvents(),
                color: Theme.of(context).colorScheme.primary,
                strokeWidth: 3.0,
                backgroundColor: Colors.transparent,
                child: filteredEvents.isEmpty
                    ? Center(
                        child: Text(
                          'No events in ${eventProvider.getFilterLabel().toLowerCase()}',
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                      )
                    : ListView.builder(
                        itemCount: filteredEvents.length,
                        itemBuilder: (context, index) {
                          final event = filteredEvents[index];
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
                                          loadingBuilder: (context, child,
                                              loadingProgress) {
                                            if (loadingProgress == null)
                                              return child;
                                            return Container(
                                              color: Colors.grey.shade800,
                                              child: SpinKitFadingCircle(
                                                color: Theme.of(context)
                                                    .colorScheme
                                                    .primary,
                                                size: 30.0,
                                              ),
                                            );
                                          },
                                          errorBuilder:
                                              (context, error, stackTrace) {
                                            // Fallback to gradient placeholder on error
                                            return Container(
                                              decoration: const BoxDecoration(
                                                  gradient: kAppGradient),
                                              child: Center(
                                                child: Icon(
                                                  Icons.videocam,
                                                  color: Colors.white
                                                      .withOpacity(0.7),
                                                  size: 32,
                                                ),
                                              ),
                                            );
                                          },
                                        )
                                      : Container(
                                          decoration: const BoxDecoration(
                                              gradient: kAppGradient),
                                          child: Center(
                                            child: Icon(
                                              Icons.videocam,
                                              color: Colors.white
                                                  .withOpacity(0.7),
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
                      ),
              ),
            ),
          ],
        );
      },
    );
  }
}