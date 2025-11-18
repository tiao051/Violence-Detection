import 'dart:async';
import 'dart:convert';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;

/// Configuration for WebRTC WHEP streaming
class WebRTCConfig {
  final String baseUrl; // e.g., "http://mediamtx:8889"
  final List<Map<String, dynamic>> iceServers;
  final Duration reconnectDelay;
  final int maxRetries;

  WebRTCConfig({
    required this.baseUrl,
    this.iceServers = const [
      {'urls': 'stun:stun.l.google.com:19302'},
      {'urls': 'stun:stun1.l.google.com:19302'},
    ],
    this.reconnectDelay = const Duration(seconds: 3),
    this.maxRetries = 5,
  });
}

/// WebRTC Streaming Service using WHEP (WebRTC HTTP Egress Protocol)
/// Handles peer connection setup, SDP offer/answer exchange with MediaMTX
class WebRTCService {
  final WebRTCConfig config;
  
  RTCPeerConnection? _peerConnection;
  RTCVideoRenderer? _remoteRenderer;
  
  String? _cameraId;
  int _connectionAttempts = 0;
  
  // Streams for state management
  final _connectionStateController = StreamController<RTCIceConnectionState>.broadcast();
  final _statsController = StreamController<List<dynamic>>.broadcast();
  
  // Callbacks
  Function(MediaStream)? onRemoteStreamReceived;
  Function(String)? onError;
  Function()? onConnectionFailed;
  
  Stream<RTCIceConnectionState> get connectionState => _connectionStateController.stream;
  Stream<List<dynamic>> get stats => _statsController.stream;
  
  WebRTCService({required this.config});

  /// Initialize WebRTC peer connection and connect to camera stream
  Future<RTCVideoRenderer> initializeStream(String cameraId) async {
    try {
      _cameraId = cameraId;
      _connectionAttempts = 0;
      
      // Create remote video renderer
      _remoteRenderer = RTCVideoRenderer();
      await _remoteRenderer!.initialize();
      
      // Establish WebRTC connection
      await _setupPeerConnection();
      
      return _remoteRenderer!;
    } catch (e) {
      onError?.call('Failed to initialize stream: $e');
      rethrow;
    }
  }

  /// Setup RTCPeerConnection and establish WHEP connection
  Future<void> _setupPeerConnection() async {
    if (_cameraId == null) throw Exception('Camera ID not set');
    
    try {
      // Create peer connection with ICE servers
      final peerConnectionFactory = await createPeerConnection(
        {
          'iceServers': config.iceServers,
        },
        {
          'sdpSemantics': 'unified-plan',
        },
      );
      
      _peerConnection = peerConnectionFactory;
      
      // Add transceiver for receiving video only
      await _peerConnection!.addTransceiver(
        kind: RTCRtpMediaType.RTCRtpMediaTypeVideo,
        init: RTCRtpTransceiverInit(
          direction: TransceiverDirection.RecvOnly,
        ),
      );
      
      // Handle incoming remote stream
      _peerConnection!.onAddStream = (MediaStream stream) {
        print('[WebRTC] Remote stream received: ${stream.id}');
        _remoteRenderer?.srcObject = stream;
        onRemoteStreamReceived?.call(stream);
      };
      
      // Monitor connection state changes
      _peerConnection!.onConnectionState = (RTCPeerConnectionState state) {
        print('[WebRTC] Peer connection state: $state');
        
        if (state == RTCPeerConnectionState.RTCPeerConnectionStateConnected) {
          _connectionAttempts = 0; // Reset retry counter on success
        }
      };
      
      // Monitor ICE connection state
      _peerConnection!.onIceConnectionState = (RTCIceConnectionState state) {
        print('[WebRTC] ICE connection state: $state');
        _connectionStateController.add(state);
        
        if (state == RTCIceConnectionState.RTCIceConnectionStateFailed ||
            state == RTCIceConnectionState.RTCIceConnectionStateClosed ||
            state == RTCIceConnectionState.RTCIceConnectionStateDisconnected) {
          if (_connectionAttempts < config.maxRetries) {
            _attemptReconnect();
          } else {
            onConnectionFailed?.call();
          }
        }
      };
      
      // Create and send SDP offer
      await _sendWhepOffer();
      
    } catch (e) {
      onError?.call('Peer connection setup failed: $e');
      rethrow;
    }
  }

  /// Create SDP offer and send to MediaMTX WHEP endpoint
  Future<void> _sendWhepOffer() async {
    if (_peerConnection == null || _cameraId == null) {
      throw Exception('Peer connection or camera ID not initialized');
    }
    
    try {
      // Create offer
      final offer = await _peerConnection!.createOffer();
      await _peerConnection!.setLocalDescription(offer);
      
      if (offer.sdp == null) {
        throw Exception('Failed to generate SDP offer');
      }
      
      print('[WebRTC] Sending WHEP offer to MediaMTX...');
      
      // Send offer to WHEP endpoint and receive answer
      final whepUrl = '${config.baseUrl}/$_cameraId/whep';
      final response = await http.post(
        Uri.parse(whepUrl),
        headers: {'Content-Type': 'application/sdp'},
        body: offer.sdp,
      ).timeout(const Duration(seconds: 10));
      
      if (response.statusCode != 201 && response.statusCode != 200) {
        throw Exception('WHEP request failed: ${response.statusCode} - ${response.body}');
      }
      
      print('[WebRTC] Received WHEP answer from MediaMTX');
      
      // Set remote description (answer)
      final answer = RTCSessionDescription(response.body, 'answer');
      await _peerConnection!.setRemoteDescription(answer);
      
      print('[WebRTC] WebRTC connection established successfully');
      
    } catch (e) {
      onError?.call('WHEP offer/answer failed: $e');
      rethrow;
    }
  }

  /// Attempt to reconnect with exponential backoff
  Future<void> _attemptReconnect() async {
    if (_connectionAttempts >= config.maxRetries) {
      onError?.call('Max reconnection attempts reached');
      return;
    }
    
    _connectionAttempts++;
    final backoffDelay = Duration(
      seconds: config.reconnectDelay.inSeconds * (_connectionAttempts * 2),
    );
    
    print('[WebRTC] Reconnecting... (attempt $_connectionAttempts/${config.maxRetries}) '
        'in ${backoffDelay.inSeconds}s');
    
    await Future.delayed(backoffDelay);
    
    try {
      await _cleanupPeerConnection();
      await _setupPeerConnection();
    } catch (e) {
      onError?.call('Reconnection failed: $e');
    }
  }

  /// Get WebRTC connection statistics
  Future<void> getStats() async {
    if (_peerConnection == null) return;
    
    try {
      final report = await _peerConnection!.getStats();
      _statsController.add(report as List<dynamic>);
      
      // Log useful stats
      for (var stat in report) {
        if (stat.type == 'inbound-rtp' && stat.values['mediaType'] == 'video') {
          print('[WebRTC Stats] '
              'Bytes received: ${stat.values['bytesReceived']}, '
              'Packets lost: ${stat.values['packetsLost']}, '
              'Frames decoded: ${stat.values['framesDecoded']}');
        }
      }
    } catch (e) {
      print('[WebRTC] Failed to get stats: $e');
    }
  }

  /// Close peer connection and clean up resources
  Future<void> _cleanupPeerConnection() async {
    if (_peerConnection != null) {
      await _peerConnection!.close();
      _peerConnection = null;
    }
  }

  /// Dispose service and clean up all resources
  Future<void> dispose() async {
    await _cleanupPeerConnection();
    
    if (_remoteRenderer != null) {
      await _remoteRenderer!.dispose();
      _remoteRenderer = null;
    }
    
    await _connectionStateController.close();
    await _statsController.close();
  }
}
