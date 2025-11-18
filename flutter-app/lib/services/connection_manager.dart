import 'dart:async';
import 'package:security_app/services/webrtc_service.dart';
import 'package:security_app/services/threat_alert_service.dart';

/// Connection status enum
enum ConnectionStatus {
  disconnected,
  connecting,
  connected,
  reconnecting,
  failed,
}

/// Unified connection manager for WebRTC streaming + Threat alerts
/// Handles lifecycle, error recovery, and statistics across both services
class ConnectionManager {
  final WebRTCService webrtcService;
  final ThreatAlertService threatAlertService;
  final String cameraId;
  
  // State management
  final _statusController = StreamController<ConnectionStatus>.broadcast();
  final _statsController = StreamController<ConnectionStats>.broadcast();
  
  ConnectionStatus _currentStatus = ConnectionStatus.disconnected;
  DateTime _lastConnectionTime = DateTime.now();
  int _totalReconnects = 0;
  
  // Getters
  Stream<ConnectionStatus> get statusStream => _statusController.stream;
  Stream<ConnectionStats> get statsStream => _statsController.stream;
  ConnectionStatus get status => _currentStatus;
  bool get isConnected => _currentStatus == ConnectionStatus.connected;
  
  // Callbacks
  Function(ConnectionStatus)? onStatusChanged;
  Function(String)? onError;
  Function()? onFullyConnected;

  ConnectionManager({
    required this.webrtcService,
    required this.threatAlertService,
    required this.cameraId,
  });

  /// Initialize both WebRTC and Threat Alert services
  Future<void> connect() async {
    if (_currentStatus != ConnectionStatus.disconnected) {
      print('[ConnectionManager] Already connecting or connected');
      return;
    }
    
    try {
      _setStatus(ConnectionStatus.connecting);
      
      print('[ConnectionManager] Starting connection for camera: $cameraId');
      
      // Setup WebRTC error handling
      webrtcService.onError = _handleWebRTCError;
      webrtcService.onConnectionFailed = _handleWebRTCConnectionFailed;
      
      // Setup Threat Alert error handling
      threatAlertService.onError = _handleThreatAlertError;
      threatAlertService.onDisconnected = _handleThreatAlertDisconnected;
      
      // Start WebRTC connection
      print('[ConnectionManager] Initializing WebRTC stream...');
      await webrtcService.initializeStream(cameraId);
      
      // Start Threat Alert connection (non-blocking, can fail without breaking video)
      print('[ConnectionManager] Initializing threat alert service...');
      try {
        await threatAlertService.connect();
      } catch (e) {
        print('[ConnectionManager] Warning: Threat alert connection failed (non-critical): $e');
        // Continue anyway - video is more important than alerts
      }
      
      _setStatus(ConnectionStatus.connected);
      _lastConnectionTime = DateTime.now();
      onFullyConnected?.call();
      
      print('[ConnectionManager] Connection established successfully');
      
      // Start stats collection
      _startStatsCollection();
      
    } catch (e) {
      _setStatus(ConnectionStatus.failed);
      final errorMsg = 'Connection failed: $e';
      print('[ConnectionManager] Error: $errorMsg');
      onError?.call(errorMsg);
      rethrow;
    }
  }

  /// Handle WebRTC errors
  void _handleWebRTCError(String error) {
    print('[ConnectionManager] WebRTC error: $error');
    onError?.call('WebRTC: $error');
    
    if (_currentStatus == ConnectionStatus.connected) {
      _setStatus(ConnectionStatus.reconnecting);
    }
  }

  /// Handle WebRTC connection failures
  void _handleWebRTCConnectionFailed() {
    print('[ConnectionManager] WebRTC connection failed (max retries)');
    _setStatus(ConnectionStatus.failed);
    onError?.call('WebRTC connection failed - max retries reached');
  }

  /// Handle Threat Alert errors
  void _handleThreatAlertError(String error) {
    print('[ConnectionManager] Threat Alert error: $error');
    // Non-critical - just log, don't affect overall connection status
    onError?.call('Threat Alert: $error');
  }

  /// Handle Threat Alert disconnection
  void _handleThreatAlertDisconnected() {
    print('[ConnectionManager] Threat Alert disconnected');
    // Non-critical - video continues
  }

  /// Set connection status and notify listeners
  void _setStatus(ConnectionStatus newStatus) {
    if (_currentStatus == newStatus) return;
    
    _currentStatus = newStatus;
    print('[ConnectionManager] Status changed: ${_currentStatus.name}');
    
    _statusController.add(newStatus);
    onStatusChanged?.call(newStatus);
    
    if (newStatus == ConnectionStatus.failed) {
      _totalReconnects++;
    }
  }

  /// Periodically collect and emit statistics
  void _startStatsCollection() {
    Timer.periodic(const Duration(seconds: 5), (timer) {
      if (!isConnected) {
        timer.cancel();
        return;
      }
      
      _collectStats();
    });
  }

  /// Collect connection statistics
  Future<void> _collectStats() async {
    try {
      // Get WebRTC stats
      await webrtcService.getStats();
      
      final stats = ConnectionStats(
        timestamp: DateTime.now(),
        webrtcConnected: isConnected,
        threatAlertConnected: threatAlertService.isConnected,
        uptime: DateTime.now().difference(_lastConnectionTime),
        totalReconnects: _totalReconnects,
      );
      
      _statsController.add(stats);
      
    } catch (e) {
      print('[ConnectionManager] Error collecting stats: $e');
    }
  }

  /// Gracefully disconnect both services
  Future<void> disconnect() async {
    print('[ConnectionManager] Disconnecting...');
    
    try {
      await webrtcService.dispose();
      await threatAlertService.disconnect();
      
      _setStatus(ConnectionStatus.disconnected);
      print('[ConnectionManager] Disconnected successfully');
      
    } catch (e) {
      print('[ConnectionManager] Error during disconnect: $e');
      onError?.call('Disconnect error: $e');
    }
  }

  /// Dispose all resources
  Future<void> dispose() async {
    await disconnect();
    
    if (!_statusController.isClosed) {
      await _statusController.close();
    }
    
    if (!_statsController.isClosed) {
      await _statsController.close();
    }
  }
}

/// Connection statistics data class
class ConnectionStats {
  final DateTime timestamp;
  final bool webrtcConnected;
  final bool threatAlertConnected;
  final Duration uptime;
  final int totalReconnects;

  ConnectionStats({
    required this.timestamp,
    required this.webrtcConnected,
    required this.threatAlertConnected,
    required this.uptime,
    required this.totalReconnects,
  });

  @override
  String toString() => 'ConnectionStats('
      'timestamp=$timestamp, '
      'webrtc=$webrtcConnected, '
      'threatAlert=$threatAlertConnected, '
      'uptime=${uptime.inSeconds}s, '
      'reconnects=$totalReconnects)';
}
