import 'dart:async';

/// Connection status enum
enum ConnectionStatus {
  disconnected,
  connecting,
  connected,
  reconnecting,
  failed,
}

/// Connection manager for streaming
/// Handles lifecycle and statistics
class ConnectionManager {
  final String cameraId;
  
  // State management
  final _statusController = StreamController<ConnectionStatus>.broadcast();
  
  ConnectionStatus _currentStatus = ConnectionStatus.disconnected;
  
  // Getters
  Stream<ConnectionStatus> get statusStream => _statusController.stream;
  ConnectionStatus get status => _currentStatus;
  bool get isConnected => _currentStatus == ConnectionStatus.connected;
  
  // Callbacks
  Function(ConnectionStatus)? onStatusChanged;
  Function(String)? onError;
  Function()? onFullyConnected;

  ConnectionManager({
    required this.cameraId,
  });

  /// Initialize streaming
  Future<void> connect() async {
    if (_currentStatus != ConnectionStatus.disconnected) {
      return;
    }
    
    try {
      _setStatus(ConnectionStatus.connecting);
      _setStatus(ConnectionStatus.connected);
      onFullyConnected?.call();
      
    } catch (e) {
      _setStatus(ConnectionStatus.failed);
      final errorMsg = 'Connection failed: $e';
      onError?.call(errorMsg);
      rethrow;
    }
  }

  /// Set connection status and notify listeners
  void _setStatus(ConnectionStatus newStatus) {
    if (_currentStatus == newStatus) return;
    
    _currentStatus = newStatus;
    
    _statusController.add(newStatus);
    onStatusChanged?.call(newStatus);
  }

  /// Gracefully disconnect
  Future<void> disconnect() async {
    try {
      _setStatus(ConnectionStatus.disconnected);
      
    } catch (e) {
      onError?.call('Disconnect error: $e');
    }
  }

  /// Dispose all resources
  Future<void> dispose() async {
    await disconnect();
    
    if (!_statusController.isClosed) {
      await _statusController.close();
    }
  }
}

/// Connection statistics data class
class ConnectionStats {
  final DateTime timestamp;
  final bool streamConnected;
  final Duration uptime;
  final int totalReconnects;

  ConnectionStats({
    required this.timestamp,
    required this.streamConnected,
    required this.uptime,
    required this.totalReconnects,
  });

  @override
  String toString() => 'ConnectionStats('
      'timestamp=$timestamp, '
      'stream=$streamConnected, '
      'uptime=${uptime.inSeconds}s, '
      'reconnects=$totalReconnects)';
}
