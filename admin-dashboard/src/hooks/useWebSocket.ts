import { useEffect, useRef, useState, useCallback } from 'react';

interface ThreatAlert {
  type: 'alert';
  camera_id: string;
  timestamp: string;
  confidence: number;
}

interface WebSocketHookResult {
  alerts: ThreatAlert[];
  isConnected: boolean;
  error: string | null;
  clearAlerts: () => void;
}

export const useWebSocket = (url: string): WebSocketHookResult => {
  const [alerts, setAlerts] = useState<ThreatAlert[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  // In browser env setTimeout returns a number, avoid NodeJS namespace
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const baseReconnectDelay = 1000; // 1 second

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  const getReconnectDelay = useCallback((attempts: number) => {
    // Exponential backoff: 1s, 2s, 4s, 8s, 16s, then stay at 16s
    return Math.min(baseReconnectDelay * Math.pow(2, attempts), 16000);
  }, []);

  const connect = useCallback(() => {
    // Don't connect if we've exceeded max attempts
    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      setError(`Failed to connect after ${maxReconnectAttempts} attempts`);
      return;
    }

    try {
      // Close existing connection if any
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      setError(null);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      // Set a connection timeout
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) {
          ws.close();
          setError('Connection timeout');
        }
      }, 5000);

      ws.onopen = () => {
        clearTimeout(connectionTimeout);
        console.log('WebSocket connected to', url);
        setIsConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0; // Reset on successful connection
      };

      ws.onmessage = (event) => {
        try {
          const data: ThreatAlert = JSON.parse(event.data);
          if (data.type === 'alert') {
            console.log('Received threat alert:', data);
            setAlerts(prev => [...prev, data]);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onclose = (event) => {
        clearTimeout(connectionTimeout);
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        wsRef.current = null;

        // Only reconnect if it wasn't a clean close
        if (event.code !== 1000) {
          reconnectAttemptsRef.current++;
          const delay = getReconnectDelay(reconnectAttemptsRef.current - 1);
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);

          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, delay);
        }
      };

      ws.onerror = (event) => {
        clearTimeout(connectionTimeout);
        console.error('WebSocket error:', event);
        setError('WebSocket connection failed');
        setIsConnected(false);
      };

    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setError('Failed to connect to WebSocket');
    }
  }, [url, getReconnectDelay]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current as number);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'Component unmounting'); // Clean close
      wsRef.current = null;
    }
    setIsConnected(false);
    reconnectAttemptsRef.current = 0;
  }, []);

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    alerts,
    isConnected,
    error,
    clearAlerts
  };
};