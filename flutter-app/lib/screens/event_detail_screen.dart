import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../models/event_model.dart';
import '../providers/event_provider.dart';
import 'package:video_player/video_player.dart';
import 'package:chewie/chewie.dart';
import '../widgets/error_widget.dart' as error_widget;

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
    // Mark event as viewed when screen loads
    context.read<EventProvider>().markEventAsViewed(widget.event.id);
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
    _chewieController.dispose();
    _controller.dispose();
    super.dispose();
  }

  /// Handles the "Report false detection" button press.
  ///
  /// Uses context.read() because we only need to call the provider method,
  /// not rebuild when state changes. Shows success/error SnackBar.
  Future<void> _handleReport() async {
    final eventProvider = context.read<EventProvider>();
    final bool success = await eventProvider.reportEvent(widget.event.id);

    if (mounted) {
      if (success) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Report sent successfully (simulated)"),
            backgroundColor: Colors.green,
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Error: ${eventProvider.reportError ?? 'Unknown'}"),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Event Details: ${widget.event.id}"),
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
                        color: Theme.of(context).colorScheme.primary,
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
                "Camera: ${widget.event.cameraName}",
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 8),
              Text(
                "Time: ${DateFormat('HH:mm:ss - dd/MM/yyyy').format(widget.event.timestamp)}",
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              
              const SizedBox(height: 32),

              // Consumer enables per-button loading state by listening to
              // the specific event ID's reporting status
              Consumer<EventProvider>(
                builder: (context, eventProvider, child) {
                  final isReporting = eventProvider.isReporting(widget.event.id);
                  final hasReportError = eventProvider.reportError != null;
                  final errorColor = Theme.of(context).colorScheme.error;
                  
                  // Disable button if already reporting or if there's a network error
                  final isDisabled = isReporting || hasReportError;

                  return Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          errorColor.withOpacity(0.8),
                          errorColor,
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(8),
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
                          // Disable button while reporting or if network error
                          onPressed: isDisabled ? null : _handleReport,
                          icon: isReporting 
                            ? const SizedBox(
                                width: 20, 
                                height: 20, 
                                child: SpinKitFadingCircle(color: Colors.white, size: 20.0)
                              ) 
                            : const Icon(Icons.report_problem),
                          label: Text(
                            isReporting 
                              ? "Sending..." 
                              : hasReportError 
                                ? "No Internet Connection"
                                : "Report false detection"
                          ),
                        ),
                        // Show error message if report failed
                        if (hasReportError)
                          Padding(
                            padding: const EdgeInsets.only(top: 8.0),
                            child: Text(
                              "Error: ${eventProvider.reportError}",
                              style: const TextStyle(
                                color: Colors.white70,
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