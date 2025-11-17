/**
 * WebSocket hook for real-time threat detection updates.
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { IWebSocketMessage } from '../types/detection';

interface IUseWebSocketOptions {
  url: string;
  onMessage?: (message: IWebSocketMessage) => void;
  onError?: (error: Event) => void;
  autoReconnect?: boolean;
  reconnectDelay?: number;
  maxRetries?: number;
}

interface IUseWebSocketReturn {
  isConnected: boolean;
  lastMessage: IWebSocketMessage | null;
  send: (data: unknown) => void;
  disconnect: () => void;
}

const DEFAULT_RECONNECT_DELAY = 3000;
const DEFAULT_MAX_RETRIES = 5;

export const useWebSocket = ({
  url,
  onMessage,
  onError,
  autoReconnect = true,
  reconnectDelay = DEFAULT_RECONNECT_DELAY,
  maxRetries = DEFAULT_MAX_RETRIES,
}: IUseWebSocketOptions): IUseWebSocketReturn => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [lastMessage, setLastMessage] = useState<IWebSocketMessage | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryCountRef = useRef<number>(0);

  const clearReconnectTimeout = useCallback((): void => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  const handleOpen = useCallback((): void => {
    setIsConnected(true);
    retryCountRef.current = 0;
    console.log('WebSocket connected successfully');
    clearReconnectTimeout();
  }, [clearReconnectTimeout]);

  const handleMessage = useCallback((event: MessageEvent): void => {
    try {
      const message = JSON.parse(event.data) as IWebSocketMessage;
      setLastMessage(message);
      onMessage?.(message);
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }, [onMessage]);

  const handleError = useCallback((error: Event): void => {
    console.error('WebSocket error occurred:', error);
    onError?.(error);
  }, [onError]);

  const scheduleReconnect = useCallback((): void => {
    if (autoReconnect && retryCountRef.current < maxRetries) {
      retryCountRef.current += 1;
      console.log(
        `Scheduling reconnection attempt ${retryCountRef.current}/${maxRetries} in ${reconnectDelay}ms`
      );
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, reconnectDelay);
    } else if (retryCountRef.current >= maxRetries) {
      console.error(
        `Maximum retry attempts (${maxRetries}) reached. Stopping reconnection.`
      );
    }
  }, [autoReconnect, maxRetries, reconnectDelay]);

  const handleClose = useCallback((): void => {
    setIsConnected(false);
    console.log('WebSocket connection closed');
    scheduleReconnect();
  }, [scheduleReconnect]);

  const connect = useCallback((): void => {
    try {
      const websocket = new WebSocket(url);

      websocket.onopen = handleOpen;
      websocket.onmessage = handleMessage;
      websocket.onerror = handleError;
      websocket.onclose = handleClose;

      websocketRef.current = websocket;
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      scheduleReconnect();
    }
  }, [url, handleOpen, handleMessage, handleError, handleClose, scheduleReconnect]);

  const disconnect = useCallback((): void => {
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const send = useCallback((data: unknown): void => {
    if (!websocketRef.current || !isConnected) {
      console.warn('Cannot send message: WebSocket is not connected');
      return;
    }

    try {
      websocketRef.current.send(JSON.stringify(data));
    } catch (error) {
      console.error('Failed to send WebSocket message:', error);
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
