import { useEffect, useRef, useState, useCallback } from 'react';

export interface WebSocketMessage {
  type: 'alert' | 'event_saved';
  camera_id: string;
  timestamp: number;
  confidence?: number;
  video_url?: string;
  snapshot?: string;
  violence?: boolean;
}

interface WebSocketHookResult {
  messages: WebSocketMessage[];
  isConnected: boolean;
  error: string | null;
  clearMessages: () => void;
}

export const useWebSocket = (url: string): WebSocketHookResult => {
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  // In development React StrictMode can mount/unmount components twice,
  // which may cause duplicate WebSocket connections. Use a simple
  // module-level singleton to avoid creating multiple underlying sockets.
  const createdByThisHookRef = useRef(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const globalAny: any = window as any;
  // In browser env setTimeout returns a number, avoid NodeJS namespace
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const baseReconnectDelay = 1000; // 1 second

  const clearMessages = useCallback(() => {
    setMessages([]);
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
      // Reuse an existing global socket if available to avoid duplicates
      if (!wsRef.current && globalAny.__global_ws && (globalAny.__global_ws.readyState === WebSocket.OPEN || globalAny.__global_ws.readyState === WebSocket.CONNECTING)) {
        wsRef.current = globalAny.__global_ws as WebSocket;
        createdByThisHookRef.current = false;
      }

      // Close existing socket only if this hook created it
      if (wsRef.current) {
        if (createdByThisHookRef.current) {
          wsRef.current.close();
          wsRef.current = null;
          createdByThisHookRef.current = false;
        } else {
          // reuse the socket instance; we'll rebind handlers below
        }
      }

      setError(null);

      if (!wsRef.current) {
        const ws = new WebSocket(url);
        wsRef.current = ws;
        // expose globally so other hook instances don't create duplicates
        globalAny.__global_ws = ws;
        createdByThisHookRef.current = true;
      }

      const ws = wsRef.current as WebSocket;

      // Set a connection timeout
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) {
          ws.close();
          setError('Connection timeout');
        }
      }, 5000);

      ws.onopen = () => {
        clearTimeout(connectionTimeout);
        setIsConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0; // Reset on successful connection
      };

      ws.onmessage = (event) => {
        try {
          const data: ThreatAlert = JSON.parse(event.data);
          if (data.type === 'alert' || data.type === 'event_saved') {
            setMessages(prev => [...prev, data]);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onclose = (event) => {
        clearTimeout(connectionTimeout);
        setIsConnected(false);
        // If this hook created the socket, clear globals when closing
        if (createdByThisHookRef.current) {
          wsRef.current = null;
          delete (globalAny as any).__global_ws;
          createdByThisHookRef.current = false;
        }

        // Only reconnect if it wasn't a clean close
        if (event.code !== 1000) {
          reconnectAttemptsRef.current++;
          const delay = getReconnectDelay(reconnectAttemptsRef.current - 1);

          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, delay);
        }
      };

      ws.onerror = (event) => {
        clearTimeout(connectionTimeout);
        setError('WebSocket connection failed');
        setIsConnected(false);
      };
    } catch (err) {
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
    messages,
    isConnected,
    error,
    clearMessages
  };
};