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
          playedColor: kAccentColor,
          handleColor: Colors.white,
          backgroundColor: kSurfaceColor,
          bufferedColor: kTextMuted,
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
            backgroundColor: kSuccessColor,
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
                "Error: ${context.read<EventProvider>().reportError ?? 'Unknown'}"),
            backgroundColor: kErrorColor,
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
              // Video player with rounded corners
              ClipRRect(
                borderRadius: BorderRadius.circular(AppRadius.md),
                child: AspectRatio(
                  aspectRatio: 16 / 9,
                  child: Builder(builder: (context) {
                    if (_isLoadingVideo) {
                      return Container(
                        color: kSurfaceColor,
                        child: const Center(
                          child: SpinKitFadingCircle(
                            color: kAccentColor,
                            size: 50.0,
                          ),
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
              ),
              const SizedBox(height: 20),

              // Alert Header
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: kErrorColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(AppRadius.md),
                  border: Border.all(color: kErrorColor.withOpacity(0.2)),
                ),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: kErrorColor.withOpacity(0.15),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.warning_amber_rounded,
                          color: kErrorColor, size: 24),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Violence Detected',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              color: kTextPrimary,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            DateFormat('EEEE, dd MMMM yyyy • HH:mm:ss')
                                .format(event.timestamp),
                            style: const TextStyle(
                                color: kTextSecondary, fontSize: 13),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              // Camera Info Card
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: kSurfaceColor,
                  borderRadius: BorderRadius.circular(AppRadius.md),
                ),
                child: Column(
                  children: [
                    _buildInfoRow(
                        Icons.videocam_outlined, 'Camera', event.cameraName),
                    const Divider(height: 24, color: kSurfaceLight),
                    _buildInfoRow(Icons.fingerprint, 'Event ID', event.id),
                    const Divider(height: 24, color: kSurfaceLight),
                    _buildInfoRow(
                      Icons.info_outline,
                      'Status',
                      event.status == 'reported_false'
                          ? 'Reported False'
                          : 'New Alert',
                      valueColor: event.status == 'reported_false'
                          ? kWarningColor
                          : kAccentColor,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
              Consumer<EventProvider>(
                builder: (context, eventProvider, child) {
                  final isReporting = eventProvider.isReporting(event.id);
                  final hasReportError = eventProvider.reportError != null;

                  // Disable button if already reported
                  if (event.status == 'reported_false') {
                    return Center(
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 8),
                        decoration: BoxDecoration(
                          color: kWarningColor.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(20),
                          border:
                              Border.all(color: kWarningColor.withOpacity(0.3)),
                        ),
                        child: const Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.check_circle_outline,
                                color: kWarningColor, size: 18),
                            SizedBox(width: 8),
                            Text(
                              'Reported as false detection',
                              style: TextStyle(
                                  color: kWarningColor,
                                  fontWeight: FontWeight.w500),
                            ),
                          ],
                        ),
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

  /// Helper to build info row in the details card
  Widget _buildInfoRow(IconData icon, String label, String value,
      {Color? valueColor}) {
    return Row(
      children: [
        Icon(icon, color: kTextMuted, size: 20),
        const SizedBox(width: 12),
        Text(
          label,
          style: const TextStyle(color: kTextMuted, fontSize: 14),
        ),
        const Spacer(),
        Flexible(
          child: Text(
            value,
            style: TextStyle(
              color: valueColor ?? kTextPrimary,
              fontSize: 14,
              fontWeight: FontWeight.w500,
            ),
            textAlign: TextAlign.right,
            overflow: TextOverflow.ellipsis,
          ),
        ),
      ],
    );
  }
}
