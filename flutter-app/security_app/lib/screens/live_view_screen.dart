import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart'; // New import
import 'package:security_app/services/camera_service.dart';

/// Screen for viewing live video from a camera.
class LiveViewScreen extends StatefulWidget {
  /// The ID of the camera to view.
  final String cameraId;

  const LiveViewScreen({super.key, required this.cameraId});

  @override
  State<LiveViewScreen> createState() => _LiveViewScreenState();
}

class _LiveViewScreenState extends State<LiveViewScreen> {
  VideoPlayerController? _videoController;
  final CameraService _cameraService = CameraService();

  bool _isLoading = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initializePlayer();
  }

  /// Initializes the player by fetching the stream URL and setting up the controller.
  Future<void> _initializePlayer() async {
    try {
      // Fetch stream URL
      final streamUrl = await _cameraService.getStreamUrl(widget.cameraId);

      // Initialize video_player controller
      _videoController = VideoPlayerController.networkUrl(Uri.parse(streamUrl))
        ..initialize().then((_) {
          if (mounted) {
            setState(() {
              _isLoading = false;
            });
            _videoController!.play(); // Auto-play
          }
        }).catchError((error) {
          if (mounted) {
            setState(() {
              _errorMessage = 'Failed to load video: $error';
              _isLoading = false;
            });
          }
        });
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = e.toString();
          _isLoading = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _videoController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Live View - ${widget.cameraId}'),
      ),
      body: Center(
        child: Builder(builder: (context) {
          // STATE 1: LOADING
          if (_isLoading) {
            return const CircularProgressIndicator();
          }

          // STATE 2: ERROR
          if (_errorMessage != null) {
            return Text('Video load error: $_errorMessage');
          }

          // STATE 3: SUCCESS
          if (_videoController != null && _videoController!.value.isInitialized) {
            return AspectRatio(
              aspectRatio: _videoController!.value.aspectRatio,
              child: VideoPlayer(_videoController!),
            );
          }

          // STATE 4: Unknown error
          return const Text('Unable to load video');
        }),
      ),
    );
  }
}