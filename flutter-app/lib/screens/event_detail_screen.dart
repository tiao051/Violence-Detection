import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import 'package:security_app/models/event_model.dart';
import 'package:security_app/providers/event_provider.dart';
import 'package:security_app/theme/app_theme.dart';
import 'package:video_player/video_player.dart';
import 'package:chewie/chewie.dart';
import 'package:security_app/widgets/error_widget.dart' as error_widget;

class EventDetailScreen extends StatefulWidget {
  final EventModel event;
  const EventDetailScreen({super.key, required this.event});

  @override
  State<EventDetailScreen> createState() => _EventDetailScreenState();
}

class _EventDetailScreenState extends State<EventDetailScreen> {
  late VideoPlayerController _controller;
  late ChewieController _chewieController;
  bool _isLoadingVideo = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      // Check if mounted just in case
      if (mounted) {
        context.read<EventProvider>().markEventAsViewedInDb(widget.event.id);
      }
    });

    _initializePlayer();
  }

  Future<void> _initializePlayer() async {
    try {
      final videoUrl = widget.event.videoUrl;
      _controller = VideoPlayerController.networkUrl(Uri.parse(videoUrl));
      await _controller.initialize();

      _chewieController = ChewieController(
        videoPlayerController: _controller,
        autoPlay: true,
        looping: false,
        allowFullScreen: true,
        allowMuting: true,
        showControlsOnInitialize: true,
        materialProgressColors: ChewieProgressColors(
          playedColor: Theme.of(context).colorScheme.primary,
          handleColor: Colors.white,
          backgroundColor: Colors.grey.shade800,
          bufferedColor: Colors.grey.shade600,
        ),
      );

      if (mounted) {
        setState(() {
          _isLoadingVideo = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = e.toString();
          _isLoadingVideo = false;
        });
      }
    }
  }

  @override
  void dispose() {
    // Check if controllers were initialized before disposing
    if (_isLoadingVideo == false && _errorMessage == null) {
      _chewieController.dispose();
      _controller.dispose();
    }
    super.dispose();
  }

  /// Handles the "Report false detection" button press.
  Future<void> _handleReport() async {
    final eventProvider = context.read<EventProvider>();

    // SỬA LỖI 3: Dùng hàm reportEvent mới từ EventProvider
    final success =
        await context.read<EventProvider>().reportEvent(widget.event.id);

    if (mounted) {
      if (success) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Report sent successfully!"),
            backgroundColor: Colors.green,
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
                "Error: ${context.read<EventProvider>().reportError ?? 'Unknown'}"),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // SỬA LỖI 1: Lấy event từ Provider thay vì widget
    // Điều này đảm bảo chúng ta có 'viewed' status mới nhất
    // (trong trường hợp optimistic update thất bại và bị revert)
    final event = context.watch<EventProvider>().events.firstWhere(
          (e) => e.id == widget.event.id,
          orElse: () => widget.event, // Fallback to original event
        );

    return Scaffold(
      appBar: AppBar(
        title: Text("Event Details: ${event.id}"),
        // Thêm icon trạng thái "đã xem"
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16.0),
            child: Icon(
              event.viewed ? Icons.visibility_off : Icons.visibility,
              color: event.viewed ? kTextMuted : kAccentColor,
            ),
          )
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              AspectRatio(
                aspectRatio: 16 / 9,
                child: Builder(builder: (context) {
                  if (_isLoadingVideo) {
                    return Center(
                      child: SpinKitFadingCircle(
                        color: kAccentColor,
                        size: 50.0,
                      ),
                    );
                  }
                  if (_errorMessage != null) {
                    return error_widget.ErrorWidget(
                      errorMessage: _errorMessage ?? "Unable to load video",
                      onRetry: () {
                        setState(() {
                          _isLoadingVideo = true;
                          _errorMessage = null;
                        });
                        _initializePlayer();
                      },
                      iconData: Icons.play_circle_outline,
                      title: "Video Not Available",
                    );
                  }
                  return Chewie(
                    controller: _chewieController,
                  );
                }),
              ),
              const SizedBox(height: 16),
              Text(
                "Camera: ${event.cameraName}",
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 8),
              Text(
                "Time: ${DateFormat('HH:mm:ss - dd/MM/yyyy').format(event.timestamp)}",
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 8),
              Text(
                "Status: ${event.status.toUpperCase()}",
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      color: event.status == 'reported_false'
                          ? Colors.orange
                          : null,
                    ),
              ),
              const SizedBox(height: 32),
              Consumer<EventProvider>(
                builder: (context, eventProvider, child) {
                  final isReporting = eventProvider.isReporting(event.id);
                  final hasReportError = eventProvider.reportError != null;

                  // Disable button if already reported
                  if (event.status == 'reported_false') {
                    return const Center(
                      child: Chip(
                        label: Text('Reported as false'),
                        backgroundColor: Colors.orange,
                      ),
                    );
                  }

                  final isDisabled = isReporting || hasReportError;

                  return Container(
                    decoration: BoxDecoration(
                      color: kErrorColor,
                      borderRadius: BorderRadius.circular(AppRadius.md),
                    ),
                    child: Column(
                      children: [
                        ElevatedButton.icon(
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size(double.infinity, 50),
                            backgroundColor: Colors.transparent,
                            foregroundColor: Colors.white,
                            shadowColor: Colors.transparent,
                            disabledBackgroundColor: Colors.transparent,
                            elevation: 0,
                          ),
                          onPressed: isDisabled ? null : _handleReport,
                          icon: isReporting
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: SpinKitFadingCircle(
                                      color: Colors.white, size: 20.0))
                              : const Icon(Icons.report_problem_outlined),
                          label: Text(isReporting
                              ? "Sending..."
                              : hasReportError
                                  ? "Report failed"
                                  : "Report false detection"),
                        ),
                        if (hasReportError)
                          Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: Text(
                              "Error: ${eventProvider.reportError}",
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.7),
                                fontSize: 12,
                              ),
                              textAlign: TextAlign.center,
                            ),
                          ),
                      ],
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
