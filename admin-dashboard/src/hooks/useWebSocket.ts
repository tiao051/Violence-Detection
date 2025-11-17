/**
 * WebSocket hook for real-time threat detection updates.
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { WebSocketMessage } from '../types/detection';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
  autoReconnect?: boolean;
  reconnectDelay?: number;
  maxRetries?: number;
}

export const useWebSocket = ({
  url,
  onMessage,
  onError,
  autoReconnect = true,
  reconnectDelay = 3000,
  maxRetries = 5,
}: UseWebSocketOptions) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryCountRef = useRef(0);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setIsConnected(true);
        retryCountRef.current = 0;
        console.log('WebSocket connected');
        // Clear any pending reconnect timeout
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
      };

      ws.onmessage = (event: MessageEvent) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = (error: Event) => {
        console.error('WebSocket error:', error);
        onError?.(error);
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket disconnected');
        
        // Attempt to reconnect with max retries limit
        if (autoReconnect && retryCountRef.current < maxRetries) {
          retryCountRef.current += 1;
          console.log(`Attempting to reconnect... (${retryCountRef.current}/${maxRetries})`);
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectDelay);
        } else if (retryCountRef.current >= maxRetries) {
          console.error(`Max retry attempts (${maxRetries}) reached. Stopping reconnection attempts.`);
        }
      };

      websocketRef.current = ws;
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      if (autoReconnect && retryCountRef.current < maxRetries) {
        retryCountRef.current += 1;
        reconnectTimeoutRef.current = setTimeout(connect, reconnectDelay);
      }
    }
  }, [url, onMessage, onError, autoReconnect, reconnectDelay, maxRetries]);

  const disconnect = useCallback(() => {
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    setIsConnected(false);
  }, []);

  const send = useCallback((data: any) => {
    if (websocketRef.current && isConnected) {
      websocketRef.current.send(JSON.stringify(data));
    }
  }, [isConnected]);

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [url]);

  return {
    isConnected,
    lastMessage,
    send,
    disconnect,
  };
};
