import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:security_app/services/webrtc_service.dart';

/// Widget that displays WebRTC video stream with RTCVideoView
class WebRTCVideoWidget extends StatefulWidget {
  final String cameraId;
  final WebRTCService webrtcService;
  final Function(String)? onError;
  final Function()? onConnected;
  
  const WebRTCVideoWidget({
    Key? key,
    required this.cameraId,
    required this.webrtcService,
    this.onError,
    this.onConnected,
  }) : super(key: key);

  @override
  State<WebRTCVideoWidget> createState() => _WebRTCVideoWidgetState();
}

class _WebRTCVideoWidgetState extends State<WebRTCVideoWidget> {
  late RTCVideoRenderer _remoteRenderer;
  bool _isConnected = false;
  bool _isLoading = true;
  String? _errorMessage;
  RTCIceConnectionState? _connectionState;

  @override
  void initState() {
    super.initState();
    _initializeWebRTC();
    _listenToConnectionState();
  }

  /// Initialize WebRTC stream
  Future<void> _initializeWebRTC() async {
    try {
      print('[WebRTCVideoWidget] Initializing WebRTC for camera: ${widget.cameraId}');
      
      _remoteRenderer = await widget.webrtcService.initializeStream(widget.cameraId);
      
      if (mounted) {
        setState(() {
          _isLoading = false;
          _isConnected = true;
        });
        widget.onConnected?.call();
      }
      
    } catch (e) {
      final errorMsg = 'Failed to initialize WebRTC: $e';
      print('[WebRTCVideoWidget] Error: $errorMsg');
      
      if (mounted) {
        setState(() {
          _isLoading = false;
          _errorMessage = errorMsg;
        });
        widget.onError?.call(errorMsg);
      }
    }
  }

  /// Listen to WebRTC connection state changes
  void _listenToConnectionState() {
    widget.webrtcService.connectionState.listen((state) {
      if (mounted) {
        setState(() {
          _connectionState = state;
          
          if (state == RTCIceConnectionState.RTCIceConnectionStateFailed ||
              state == RTCIceConnectionState.RTCIceConnectionStateClosed ||
              state == RTCIceConnectionState.RTCIceConnectionStateDisconnected) {
            _isConnected = false;
            _errorMessage = 'Connection ${state.name}';
          } else if (state == RTCIceConnectionState.RTCIceConnectionStateConnected ||
                     state == RTCIceConnectionState.RTCIceConnectionStateCompleted) {
            _isConnected = true;
            _errorMessage = null;
          }
        });
      }
    });
  }

  @override
  void dispose() {
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Video display
        if (!_isLoading && _isConnected && _errorMessage == null)
          RTCVideoView(
            _remoteRenderer,
            objectFit: RTCVideoViewObjectFit.RTCVideoViewObjectFitContain,
            mirror: false,
          )
        else
          // Loading state
          Container(
            color: Colors.black,
            child: Center(
              child: _isLoading
                  ? const CircularProgressIndicator()
                  : Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.videocam_off,
                          color: Colors.red,
                          size: 48,
                        ),
                        const SizedBox(height: 16),
                        Text(
                          _errorMessage ?? 'Disconnected',
                          textAlign: TextAlign.center,
                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                color: Colors.white,
                              ),
                        ),
                      ],
                    ),
            ),
          ),
        
        // Connection status indicator
        Positioned(
          top: 16,
          right: 16,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: _isConnected ? Colors.green : Colors.red,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                SizedBox(
                  width: 8,
                  height: 8,
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  _isConnected ? 'Connected' : 'Disconnected',
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
        
        // Connection stats (optional, development only)
        if (_connectionState != null)
          Positioned(
            bottom: 16,
            left: 16,
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.6),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                'State: ${_connectionState?.name ?? "unknown"}',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                ),
              ),
            ),
          ),
      ],
    );
  }
}
