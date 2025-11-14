import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:video_player/video_player.dart';
import 'package:chewie/chewie.dart';
import 'package:security_app/services/camera_service.dart';
import 'package:security_app/widgets/error_widget.dart' as error_widget;

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
  ChewieController? _chewieController;
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
            // Create Chewie controller
            _chewieController = ChewieController(
              videoPlayerController: _videoController!,
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

            setState(() {
              _isLoading = false;
            });
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
    _chewieController?.dispose();
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
            return SpinKitFadingCircle(
              color: Theme.of(context).colorScheme.primary,
              size: 50.0,
            );
          }

          // STATE 2: ERROR
          if (_errorMessage != null) {
            return error_widget.ErrorWidget(
              errorMessage: _errorMessage ?? "Unable to load live stream",
              onRetry: () {
                setState(() {
                  _isLoading = true;
                  _errorMessage = null;
                });
                _initializePlayer();
              },
              iconData: Icons.live_tv_rounded,
              title: "Live Stream Unavailable",
            );
          }

          // STATE 3: SUCCESS
          if (_videoController != null && _videoController!.value.isInitialized && _chewieController != null) {
            return AspectRatio(
              aspectRatio: _videoController!.value.aspectRatio,
              child: Chewie(
                controller: _chewieController!,
              ),
            );
          }

          // STATE 4: Unknown error
          return const Text('Unable to load video');
        }),
      ),
    );
  }
}