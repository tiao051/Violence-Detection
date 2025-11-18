import 'dart:async';
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

/// Model for threat/violence detection event
class ThreatEvent {
  final String cameraId;
  final bool violence;
  final double confidence;
  final DateTime timestamp;
  final Map<String, dynamic> metadata; // Additional detection data

  ThreatEvent({
    required this.cameraId,
    required this.violence,
    required this.confidence,
    required this.timestamp,
    this.metadata = const {},
  });

  factory ThreatEvent.fromJson(Map<String, dynamic> json) {
    return ThreatEvent(
      cameraId: json['camera_id'] as String,
      violence: json['violence'] as bool,
      confidence: (json['confidence'] as num).toDouble(),
      timestamp: DateTime.fromMillisecondsSinceEpoch(
        (json['timestamp'] as num).toInt() * 1000,
      ),
      metadata: json['metadata'] as Map<String, dynamic>? ?? {},
    );
  }

  @override
  String toString() => 'ThreatEvent('
      'cameraId=$cameraId, '
      'violence=$violence, '
      'confidence=$confidence, '
      'timestamp=$timestamp)';
}

/// Configuration for Threat Alert Service
class ThreatAlertConfig {
  final String wsUrl; // e.g., "ws://backend:8000/ws/threats"
  final String? authToken; // JWT token for authentication
  final Duration reconnectDelay;
  final int maxRetries;

  ThreatAlertConfig({
    required this.wsUrl,
    this.authToken,
    this.reconnectDelay = const Duration(seconds: 3),
    this.maxRetries = 5,
  });
}

/// Service for receiving real-time threat/violence detection alerts via WebSocket
class ThreatAlertService {
  final ThreatAlertConfig config;
  
  WebSocketChannel? _channel;
  StreamController<ThreatEvent> _threatController = 
    StreamController<ThreatEvent>.broadcast();
  
  int _reconnectionAttempts = 0;
  bool _isConnected = false;
  
  // Getters
  Stream<ThreatEvent> get threatStream => _threatController.stream;
  bool get isConnected => _isConnected;
  
  // Callbacks
  Function()? onConnected;
  Function()? onDisconnected;
  Function(String)? onError;

  ThreatAlertService({required this.config});

  /// Connect to WebSocket and listen for threat alerts
  Future<void> connect() async {
    try {
      print('[ThreatAlertService] Connecting to threat alerts at ${config.wsUrl}...');
      
      _channel = WebSocketChannel.connect(Uri.parse(config.wsUrl));
      
      // Send authentication token if available
      if (config.authToken != null && config.authToken!.isNotEmpty) {
        _sendAuthMessage();
      }
      
      _isConnected = true;
      _reconnectionAttempts = 0;
      onConnected?.call();
      
      print('[ThreatAlertService] Connected successfully');
      
      // Listen to incoming messages
      _listenToMessages();
      
    } catch (e) {
      final errorMsg = 'Failed to connect to threat alerts: $e';
      print('[ThreatAlertService] Error: $errorMsg');
      onError?.call(errorMsg);
      
      // Attempt reconnection
      if (_reconnectionAttempts < config.maxRetries) {
        _attemptReconnect();
      }
    }
  }

  /// Send authentication message with JWT token
  void _sendAuthMessage() {
    try {
      final authMsg = jsonEncode({
        'type': 'auth',
        'token': config.authToken,
      });
      _channel?.sink.add(authMsg);
      print('[ThreatAlertService] Auth message sent');
    } catch (e) {
      print('[ThreatAlertService] Failed to send auth: $e');
    }
  }

  /// Listen to incoming WebSocket messages
  void _listenToMessages() {
    _channel?.stream.listen(
      (message) {
        try {
          final data = jsonDecode(message);
          
          if (data['type'] == 'threat_detection') {
            final threat = ThreatEvent.fromJson(data);
            print('[ThreatAlertService] Threat received: $threat');
            _threatController.add(threat);
          } else if (data['type'] == 'auth_success') {
            print('[ThreatAlertService] Authentication successful');
          } else if (data['type'] == 'auth_error') {
            print('[ThreatAlertService] Authentication failed: ${data['message']}');
            onError?.call('Authentication failed');
          }
        } catch (e) {
          print('[ThreatAlertService] Failed to parse message: $e');
        }
      },
      onError: (error) {
        final errorMsg = 'WebSocket error: $error';
        print('[ThreatAlertService] $errorMsg');
        onError?.call(errorMsg);
        _handleDisconnect();
      },
      onDone: () {
        print('[ThreatAlertService] WebSocket closed');
        _handleDisconnect();
      },
    );
  }

  /// Handle disconnection and attempt reconnection
  void _handleDisconnect() {
    _isConnected = false;
    onDisconnected?.call();
    
    if (_reconnectionAttempts < config.maxRetries) {
      _attemptReconnect();
    }
  }

  /// Attempt to reconnect with exponential backoff
  Future<void> _attemptReconnect() async {
    _reconnectionAttempts++;
    final backoffDelay = Duration(
      seconds: config.reconnectDelay.inSeconds * (_reconnectionAttempts * 2),
    );
    
    print('[ThreatAlertService] Reconnecting... '
        '(attempt $_reconnectionAttempts/${config.maxRetries}) '
        'in ${backoffDelay.inSeconds}s');
    
    await Future.delayed(backoffDelay);
    
    try {
      await connect();
    } catch (e) {
      print('[ThreatAlertService] Reconnection attempt $_reconnectionAttempts failed: $e');
      if (_reconnectionAttempts < config.maxRetries) {
        _attemptReconnect();
      } else {
        onError?.call('Max reconnection attempts reached');
      }
    }
  }

  /// Filter threats by camera ID
  Stream<ThreatEvent> filterByCameraId(String cameraId) {
    return threatStream.where((threat) => threat.cameraId == cameraId);
  }

  /// Filter only violence threats
  Stream<ThreatEvent> violenceOnly() {
    return threatStream.where((threat) => threat.violence);
  }

  /// Filter threats by camera ID and violence
  Stream<ThreatEvent> filterViolenceByCamera(String cameraId) {
    return threatStream.where((threat) => threat.cameraId == cameraId && threat.violence);
  }

  /// Disconnect and clean up resources
  Future<void> disconnect() async {
    print('[ThreatAlertService] Disconnecting...');
    
    _reconnectionAttempts = config.maxRetries; // Prevent auto-reconnect
    
    try {
      await _channel?.sink.close();
      _isConnected = false;
      onDisconnected?.call();
    } catch (e) {
      print('[ThreatAlertService] Error during disconnect: $e');
    }
  }

  /// Dispose service and clean up resources
  Future<void> dispose() async {
    print('[ThreatAlertService] Disposing...');
    
    await disconnect();
    
    if (!_threatController.isClosed) {
      await _threatController.close();
    }
  }
}
