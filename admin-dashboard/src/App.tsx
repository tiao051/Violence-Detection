/**
 * App - Main application component for violence detection monitoring dashboard.
 */

import React, { useCallback, useMemo, useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import CameraGrid from './components/CameraGrid';
import AlertBanner from './components/AlertBanner';
import { IWebSocketMessage, ICameraStatus } from './types/detection';
import './index.css';

interface IThreatsState {
  [cameraId: string]: ICameraStatus;
}

const CAMERA_IDS = ['cam1', 'cam2', 'cam3', 'cam4'] as const;
const BACKEND_PORT = 8000;

const createInitialThreatsState = (): IThreatsState => {
  return CAMERA_IDS.reduce((acc, cameraId) => {
    acc[cameraId] = {
      camera_id: cameraId,
      violence: false,
      confidence: 0,
    };
    return acc;
  }, {} as IThreatsState);
};

const getWebSocketUrl = (): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//localhost:${BACKEND_PORT}/ws/threats`;
};

const App: React.FC = () => {
  const [threats, setThreats] = useState<IThreatsState>(createInitialThreatsState());
  const [latestMessage, setLatestMessage] = useState<IWebSocketMessage | null>(null);

  const wsUrl = useMemo(() => getWebSocketUrl(), []);

  const handleThreatDetection = useCallback((message: IWebSocketMessage): void => {
    if (!message.camera_id) {
      return;
    }

    const cameraId = message.camera_id;
    setThreats((prevThreats) => ({
      ...prevThreats,
      [cameraId]: {
        camera_id: cameraId,
        violence: message.violence ?? false,
        confidence: message.confidence ?? 0,
        timestamp: message.timestamp,
      },
    }));
  }, []);

  const handleThreatStatus = useCallback((message: IWebSocketMessage): void => {
    if (message.threats) {
      setThreats(message.threats);
    }
  }, []);

  const handleWebSocketMessage = useCallback((message: IWebSocketMessage): void => {
    setLatestMessage(message);

    if (message.type === 'threat_detection') {
      handleThreatDetection(message);
    } else if (message.type === 'threat_status') {
      handleThreatStatus(message);
    }
  }, [handleThreatDetection, handleThreatStatus]);

  const handleWebSocketError = useCallback((error: Event): void => {
    console.error('WebSocket connection error:', error);
  }, []);

  const { isConnected } = useWebSocket({
    url: wsUrl,
    onMessage: handleWebSocketMessage,
    onError: handleWebSocketError,
  });

  const activeThreatCount = useMemo(() => {
    return Object.values(threats).filter((threat) => threat.violence).length;
  }, [threats]);

  const lastUpdateTime = useMemo(() => {
    if (!latestMessage?.timestamp) {
      return 'N/A';
    }
    return new Date(latestMessage.timestamp * 1000).toLocaleTimeString();
  }, [latestMessage]);

  const connectionStatusClass = useMemo(() => {
    return isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500';
  }, [isConnected]);

  const connectionStatusText = useMemo(() => {
    return isConnected ? 'Connected' : 'Disconnected';
  }, [isConnected]);

  return (
    <div className="w-screen h-screen bg-gray-950 flex flex-col text-white">
      <div className="bg-gray-900 border-b border-gray-800 p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-white">Violence Detection Dashboard</h1>
            <p className="text-gray-400 text-sm mt-1">
              Real-time monitoring of {CAMERA_IDS.length} cameras
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div
              className={`w-3 h-3 rounded-full ${connectionStatusClass}`}
              aria-label={`Connection status: ${connectionStatusText}`}
            />
            <span className="text-sm">{connectionStatusText}</span>
          </div>
        </div>
      </div>

      <div className="bg-gray-900 border-b border-gray-800 px-4 py-3">
        <AlertBanner message={latestMessage} />
      </div>

      <div className="flex-1 overflow-auto p-4">
        <div className="max-w-7xl mx-auto h-full">
          <CameraGrid threats={threats} />
        </div>
      </div>

      <div className="bg-gray-900 border-t border-gray-800 p-3 text-center text-sm text-gray-400">
        <p>
          {activeThreatCount} threat(s) detected | Last update: {lastUpdateTime}
        </p>
      </div>
    </div>
  );
};

export default App;
