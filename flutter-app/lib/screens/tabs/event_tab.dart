import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import 'package:security_app/providers/event_provider.dart';
import 'package:security_app/theme/app_theme.dart';
import 'package:security_app/widgets/error_widget.dart' as error_widget;
import 'package:security_app/widgets/empty_state_widget.dart';

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
          return const Center(
            child: SpinKitFadingCircle(
              color: kAccentColor,
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
                      selected: eventProvider.dateFilter == DateFilter.thisWeek,
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
                color: kAccentColor,
                strokeWidth: 3.0,
                backgroundColor: kSurfaceColor,
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
                            // === SỬA LỖI 1: Thêm màu nền (tint) ===
                            color: !event.viewed
                                ? Theme.of(context)
                                    .colorScheme
                                    .primary
                                    .withOpacity(0.05)
                                : null,
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
                                              color: kSurfaceColor,
                                              child: const Center(
                                                child: SpinKitFadingCircle(
                                                  color: kAccentColor,
                                                  size: 24.0,
                                                ),
                                              ),
                                            );
                                          },
                                          errorBuilder:
                                              (context, error, stackTrace) {
                                            return Container(
                                              color: kSurfaceColor,
                                              child: Center(
                                                child: Icon(
                                                  Icons.videocam_outlined,
                                                  color: kTextMuted,
                                                  size: 32,
                                                ),
                                              ),
                                            );
                                          },
                                        )
                                      : Container(
                                          color: kSurfaceColor,
                                          child: Center(
                                            child: Icon(
                                              Icons.videocam_outlined,
                                              color: kTextMuted,
                                              size: 32,
                                            ),
                                          ),
                                        ),
                                ),
                              ),
                              // === SỬA LỖI 2: In đậm tiêu đề ===
                              title: Text(
                                'Detected at ${event.cameraName}',
                                style: TextStyle(
                                  fontWeight: !event.viewed
                                      ? FontWeight.bold
                                      : FontWeight.normal,
                                ),
                              ),
                              subtitle: Row(
                                children: [
                                  Text(formattedTime),
                                  if (event.status == 'reported_false') ...[
                                    const SizedBox(width: 8),
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                          horizontal: 6, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: kWarningColor.withOpacity(0.15),
                                        borderRadius: BorderRadius.circular(4),
                                      ),
                                      child: const Text(
                                        'Reported',
                                        style: TextStyle(
                                          color: kWarningColor,
                                          fontSize: 10,
                                          fontWeight: FontWeight.w600,
                                        ),
                                      ),
                                    ),
                                  ],
                                ],
                              ),
                              // === SỬA LỖI 3: Thêm "chấm chưa đọc" ===
                              trailing: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  if (!event.viewed)
                                    Padding(
                                      padding:
                                          const EdgeInsets.only(right: 8.0),
                                      child: Icon(
                                        Icons.circle,
                                        size: 10,
                                        color: kAccentColor,
                                      ),
                                    ),
                                  const Icon(Icons.chevron_right), // Mũi tên cũ
                                ],
                              ),
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
