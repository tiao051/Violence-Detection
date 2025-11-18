import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:security_app/providers/auth_provider.dart';
import 'package:security_app/services/camera_service.dart';
import 'package:security_app/services/webrtc_service.dart';
import 'package:security_app/services/threat_alert_service.dart';
import 'package:security_app/widgets/webrtc_video_widget.dart';

/// Screen for viewing live video from a camera with real-time threat detection overlay.
class LiveViewScreen extends StatefulWidget {
  /// The ID of the camera to view.
  final String cameraId;

  const LiveViewScreen({super.key, required this.cameraId});

  @override
  State<LiveViewScreen> createState() => _LiveViewScreenState();
}

class _LiveViewScreenState extends State<LiveViewScreen> {
  late WebRTCService _webrtcService;
  late ThreatAlertService _threatAlertService;
  final CameraService _cameraService = CameraService();

  bool _isLoading = true;
  String? _errorMessage;
  bool _threatDetected = false;
  double _threatConfidence = 0.0;
  DateTime? _lastThreatTime;

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  /// Initialize WebRTC and Threat Alert services
  Future<void> _initializeServices() async {
    try {
      // Get access token from AuthProvider
      final authProvider = context.read<AuthProvider>();
      final accessToken = authProvider.accessToken;

      if (accessToken == null || accessToken.isEmpty) {
        throw Exception('Not authenticated. Access token is missing.');
      }

      // Get stream URL from backend
      final streamUrl = await _cameraService.getStreamUrl(
        widget.cameraId,
        accessToken: accessToken,
      );

      print('[LiveViewScreen] Stream URL received: $streamUrl');

      // Initialize WebRTC service with stream URL
      final webrtcConfig = WebRTCConfig(
        baseUrl: streamUrl,  // Use actual stream URL from backend
        iceServers: const [
          {'urls': 'stun:stun.l.google.com:19302'},
          {'urls': 'stun:stun1.l.google.com:19302'},
        ],
      );
      
      _webrtcService = WebRTCService(config: webrtcConfig);
      
      // Setup WebRTC error callbacks
      _webrtcService.onError = (error) {
        print('[LiveViewScreen] WebRTC Error: $error');
        if (mounted) {
          setState(() => _errorMessage = error);
        }
      };
      
      _webrtcService.onConnectionFailed = () {
        print('[LiveViewScreen] WebRTC connection failed');
        if (mounted) {
          setState(() => _errorMessage = 'Connection failed - max retries reached');
        }
      };

      // Initialize Threat Alert service with JWT token
      final threatConfig = ThreatAlertConfig(
        wsUrl: 'ws://localhost:8000/ws/threats',  // TODO: Update with actual backend URL
        authToken: accessToken,  // Use JWT token from AuthProvider
      );
      
      _threatAlertService = ThreatAlertService(config: threatConfig);
      
      // Setup threat alert callbacks
      _threatAlertService.onConnected = () {
        print('[LiveViewScreen] Threat alert service connected');
      };
      
      _threatAlertService.onError = (error) {
        print('[LiveViewScreen] Threat alert error: $error');
      };
      
      // Listen to threat events for this camera
      _threatAlertService.filterByCameraId(widget.cameraId).listen(
        (threat) {
          print('[LiveViewScreen] Threat received: $threat');
          if (mounted) {
            setState(() {
              _threatDetected = threat.violence;
              _threatConfidence = threat.confidence;
              _lastThreatTime = threat.timestamp;
            });
            
            // Show threat notification
            if (threat.violence) {
              _showThreatNotification(threat);
            }
          }
        },
        onError: (error) {
          print('[LiveViewScreen] Threat stream error: $error');
        },
      );

      // Connect threat alert service
      await _threatAlertService.connect();
      
      // Update UI after initialization
      if (mounted) {
        setState(() => _isLoading = false);
      }
      
    } catch (e) {
      final errorMsg = 'Failed to initialize services: $e';
      print('[LiveViewScreen] Error: $errorMsg');
      
      if (mounted) {
        setState(() {
          _isLoading = false;
          _errorMessage = errorMsg;
        });
      }
    }
  }

  /// Show threat detection notification
  void _showThreatNotification(ThreatEvent threat) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.warning, color: Colors.white),
            const SizedBox(width: 8),
            Expanded(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '⚠️ VIOLENCE DETECTED',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                    ),
                  ),
                  Text(
                    'Confidence: ${(threat.confidence * 100).toStringAsFixed(1)}%',
                    style: const TextStyle(fontSize: 12),
                  ),
                ],
              ),
            ),
          ],
        ),
        backgroundColor: Colors.red,
        duration: const Duration(seconds: 5),
        behavior: SnackBarBehavior.floating,
        margin: const EdgeInsets.all(16),
      ),
    );
  }

  @override
  void dispose() {
    _webrtcService.dispose();
    _threatAlertService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Live View - ${widget.cameraId}'),
        elevation: 0,
      ),
      body: Stack(
        children: [
          // Main content
          Center(
            child: _isLoading
                ? const CircularProgressIndicator()
                : _errorMessage != null
                    ? _buildErrorWidget()
                    : _buildVideoWidget(),
          ),

          // Threat detection overlay
          if (_threatDetected)
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.red.shade700,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.red.withOpacity(0.5),
                      blurRadius: 10,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.warning, color: Colors.white, size: 24),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                'VIOLENCE DETECTED',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              Text(
                                'Confidence: ${(_threatConfidence * 100).toStringAsFixed(1)}%',
                                style: TextStyle(
                                  color: Colors.white.withOpacity(0.9),
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: _threatConfidence,
                        minHeight: 4,
                        backgroundColor: Colors.white.withOpacity(0.3),
                        valueColor: AlwaysStoppedAnimation<Color>(
                          Colors.yellow.shade700,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

          // Threat history (optional)
          if (_lastThreatTime != null)
            Positioned(
              bottom: 16,
              right: 16,
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.7),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: _threatDetected ? Colors.red : Colors.grey,
                    width: 2,
                  ),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Last Alert',
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: 10,
                      ),
                    ),
                    Text(
                      _formatTime(_lastThreatTime!),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  /// Build video widget with WebRTC stream
  Widget _buildVideoWidget() {
    return WebRTCVideoWidget(
      cameraId: widget.cameraId,
      webrtcService: _webrtcService,
      onError: (error) {
        setState(() => _errorMessage = error);
      },
      onConnected: () {
        print('[LiveViewScreen] WebRTC connected');
      },
    );
  }

  /// Build error widget with retry button
  Widget _buildErrorWidget() {
    return Container(
      color: Colors.black,
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.videocam_off,
              color: Colors.red,
              size: 64,
            ),
            const SizedBox(height: 16),
            Text(
              'Stream Error',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    color: Colors.white,
                  ),
            ),
            const SizedBox(height: 8),
            Text(
              _errorMessage ?? 'Unknown error',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.white70,
                  ),
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: () {
                setState(() {
                  _isLoading = true;
                  _errorMessage = null;
                });
                _initializeServices();
              },
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }

  /// Format time as HH:MM:SS
  String _formatTime(DateTime time) {
    return '${time.hour.toString().padLeft(2, '0')}:'
           '${time.minute.toString().padLeft(2, '0')}:'
           '${time.second.toString().padLeft(2, '0')}';
  }
}