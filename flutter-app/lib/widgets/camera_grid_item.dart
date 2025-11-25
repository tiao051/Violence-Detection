import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';
import 'package:provider/provider.dart';
import 'package:security_app/models/camera_model.dart';
import 'package:security_app/providers/auth_provider.dart';
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
  final CameraService _cameraService = CameraService();
  
  bool _isLoading = true;
  bool _hasError = false;

  @override
  void initState() {
    super.initState();
    _player = Player();
    _controller = VideoController(_player, configuration: const VideoControllerConfiguration(enableHardwareAcceleration: true));
    _initializeStream();
  }

  Future<void> _initializeStream() async {
    if (!mounted) return;
    
    try {
      final authProvider = context.read<AuthProvider>();
      final accessToken = authProvider.accessToken;

      if (accessToken == null) {
        throw Exception('No access token');
      }

      // Get RTSP URL
      final streamUrl = await _cameraService.getStreamUrl(
        widget.camera.id,
        accessToken: accessToken,
      );

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
      child: Card(
        clipBehavior: Clip.antiAlias,
        elevation: 4,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: Stack(
          fit: StackFit.expand,
          children: [
            // Video Layer
            if (_isLoading)
              const Center(child: CircularProgressIndicator(strokeWidth: 2))
            else if (_hasError)
              Container(
                color: Colors.black87,
                child: const Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.videocam_off, color: Colors.red, size: 32),
                    SizedBox(height: 4),
                    Text(
                      'Offline',
                      style: TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                  ],
                ),
              )
            else
              Video(
                controller: _controller,
                fit: BoxFit.cover,
                controls: NoVideoControls, // Hide controls for grid view
              ),

            // Overlay Layer (Gradient + Name)
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.bottomCenter,
                    end: Alignment.topCenter,
                    colors: [
                      Colors.black.withOpacity(0.8),
                      Colors.transparent,
                    ],
                  ),
                ),
                child: Row(
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: _hasError ? Colors.red : Colors.green,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        widget.camera.name,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          shadows: [
                            Shadow(
                              offset: Offset(0, 1),
                              blurRadius: 2,
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
            ),
          ],
        ),
      ),
    );
  }
}
