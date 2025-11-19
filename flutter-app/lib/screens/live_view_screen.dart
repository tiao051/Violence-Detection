import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';       // Import mới
import 'package:media_kit_video/media_kit_video.dart'; // Import mới
import 'package:provider/provider.dart';
import 'package:security_app/providers/auth_provider.dart';
import 'package:security_app/services/camera_service.dart';

class LiveViewScreen extends StatefulWidget {
  final String cameraId;

  const LiveViewScreen({super.key, required this.cameraId});

  @override
  State<LiveViewScreen> createState() => _LiveViewScreenState();
}

class _LiveViewScreenState extends State<LiveViewScreen> {
  // Media Kit Controllers
  late final Player _player;
  late final VideoController _controller;

  final CameraService _cameraService = CameraService();

  bool _isLoading = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    
    // 1. Khởi tạo Player
    _player = Player();
    _controller = VideoController(_player);

    _initializeServices();
  }

  Future<void> _initializeServices() async {
    try {
      final authProvider = context.read<AuthProvider>();
      final accessToken = authProvider.accessToken;

      if (accessToken == null || accessToken.isEmpty) {
        throw Exception('Authentication failed: Access token is missing.');
      }

      // 2. Lấy URL RTSP từ Backend
      final streamUrl = await _cameraService.getStreamUrl(
        widget.cameraId,
        accessToken: accessToken,
      );

      debugPrint('[LiveViewScreen] RTSP Stream URL: $streamUrl');

      // 3. Mở Video với cấu hình TCP (cho ADB)
      await _player.open(
        Media(
          streamUrl,
          // QUAN TRỌNG: Ép MPV chạy RTSP qua TCP
          extras: {
            'rtsp-transport': 'tcp', 
          },
        ),
        play: true, // Tự động phát
      );
      
      if (mounted) {
        setState(() => _isLoading = false);
      }
      
    } catch (e) {
      debugPrint('[LiveViewScreen] Error: $e');
      if (mounted) {
        setState(() {
          _isLoading = false;
          _errorMessage = e.toString();
        });
      }
    }
  }

  @override
  void dispose() {
    // Giải phóng tài nguyên MediaKit
    _player.dispose(); 
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Live View - ${widget.cameraId}')),
      body: Stack(
        children: [
          // 1. Video Layer
          Center(
            child: _isLoading
                ? const CircularProgressIndicator()
                : _errorMessage != null
                    ? _buildErrorWidget()
                    : Video(controller: _controller), // Widget của MediaKit
          ),
        ],
      ),
    );
  }

  Widget _buildErrorWidget() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error, color: Colors.red, size: 50),
          const SizedBox(height: 10),
          Text(_errorMessage ?? 'Unknown Error', textAlign: TextAlign.center),
          ElevatedButton(
            onPressed: () {
              setState(() {
                _isLoading = true;
                _errorMessage = null;
              });
              _initializeServices();
            },
            child: const Text('Retry'),
          )
        ],
      ),
    );
  }
}