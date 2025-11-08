import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import 'package:security_app/models/event_model.dart';
import 'package:security_app/providers/event_provider.dart';
import 'package:video_player/video_player.dart';

class EventDetailScreen extends StatefulWidget {
  final EventModel event;
  const EventDetailScreen({super.key, required this.event});

  @override
  State<EventDetailScreen> createState() => _EventDetailScreenState();
}

class _EventDetailScreenState extends State<EventDetailScreen> {
  late VideoPlayerController _controller;
  bool _isLoadingVideo = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initializePlayer();
  }

  Future<void> _initializePlayer() async {
    try {
      final videoUrl = widget.event.videoUrl;
      _controller = VideoPlayerController.networkUrl(Uri.parse(videoUrl));
      await _controller.initialize();
      await _controller.play();
      await _controller.setVolume(1.0);

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
                    return const Center(child: CircularProgressIndicator());
                  }
                  if (_errorMessage != null) {
                    return Center(child: Text("Video load error: $_errorMessage"));
                  }
                  return VideoPlayer(_controller);
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

                  return ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      minimumSize: const Size(double.infinity, 50),
                      backgroundColor: Colors.red.shade700,
                      foregroundColor: Colors.white,
                      disabledBackgroundColor: Colors.red.shade300,
                    ),
                    // Disable button while reporting to prevent duplicate submissions
                    onPressed: isReporting ? null : _handleReport,
                    icon: isReporting 
                      ? const SizedBox(
                          width: 20, 
                          height: 20, 
                          child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)
                        ) 
                      : const Icon(Icons.report_problem),
                    label: Text(isReporting ? "Sending..." : "Report false detection"),
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