import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';
import 'package:security_app/models/camera_model.dart';
import 'package:security_app/services/camera_service.dart';

class CameraGridItem extends StatefulWidget {
  final CameraModel camera;
  final VoidCallback onTap;

  const CameraGridItem({
    super.key,
    required this.camera,
    required this.onTap,
  });

  @override
  State<CameraGridItem> createState() => _CameraGridItemState();
}

class _CameraGridItemState extends State<CameraGridItem> {
  late final Player _player;
  late final VideoController _controller;
  // CameraService removed as it is unused

  bool _isLoading = true;
  bool _hasError = false;

  @override
  void initState() {
    super.initState();
    _player = Player();
    _controller = VideoController(_player,
        configuration: const VideoControllerConfiguration(
            enableHardwareAcceleration: true));
    _initializeStream();
  }

  Future<void> _initializeStream() async {
    if (!mounted) return;

    try {
      // Use streamUrl directly from the model (already populated by backend with HLS URL)
      final streamUrl = widget.camera.streamUrl;

      if (streamUrl.isEmpty) {
        throw Exception('Stream URL is empty');
      }

      if (!mounted) return;

      // Open stream with low latency settings
      await _player.open(
        Media(
          streamUrl,
          extras: {
            'rtsp-transport': 'tcp',
            'analyzeduration': '500000', // Lower buffer for faster start
            'probesize': '500000',
          },
        ),
        play: true,
      );

      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = false;
        });
      }
    } catch (e) {
      debugPrint('Error loading stream for ${widget.camera.name}: $e');
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
        });
      }
    }
  }

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: widget.onTap,
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.3),
              blurRadius: 8,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: Stack(
            fit: StackFit.expand,
            children: [
              // Video Layer
              if (_isLoading)
                Container(
                  color: Colors.black87,
                  child: const Center(
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white24,
                    ),
                  ),
                )
              else if (_hasError)
                Container(
                  color: const Color(0xFF1A1A1A),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.videocam_off,
                          color: Colors.white.withOpacity(0.3), size: 32),
                      const SizedBox(height: 8),
                      Text(
                        'Offline',
                        style: TextStyle(
                            color: Colors.white.withOpacity(0.5), fontSize: 12),
                      ),
                    ],
                  ),
                )
              else
                Video(
                  controller: _controller,
                  fit: BoxFit.cover,
                  controls: NoVideoControls,
                ),

              // Gradient Overlay
              Positioned.fill(
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        Colors.black.withOpacity(0.2),
                        Colors.transparent,
                        Colors.transparent,
                        Colors.black.withOpacity(0.8),
                      ],
                      stops: const [0.0, 0.2, 0.6, 1.0],
                    ),
                  ),
                ),
              ),

              // LIVE Badge
              if (!_hasError && !_isLoading)
                Positioned(
                  top: 8,
                  right: 8,
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.red.withOpacity(0.9),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.circle, size: 6, color: Colors.white),
                        SizedBox(width: 4),
                        Text(
                          'LIVE',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 0.5,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

              // Camera Name
              Positioned(
                bottom: 12,
                left: 12,
                right: 12,
                child: Row(
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: _hasError ? Colors.red : const Color(0xFF00FF94),
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: (_hasError
                                    ? Colors.red
                                    : const Color(0xFF00FF94))
                                .withOpacity(0.5),
                            blurRadius: 6,
                            spreadRadius: 1,
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        widget.camera.name,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          shadows: [
                            Shadow(
                              offset: Offset(0, 1),
                              blurRadius: 4,
                              color: Colors.black,
                            ),
                          ],
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
