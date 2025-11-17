/**
 * App - Main application component for violence detection monitoring dashboard.
 */

import React, { useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import CameraGrid from './components/CameraGrid';
import AlertBanner from './components/AlertBanner';
import { WebSocketMessage, CameraStatus } from './types/detection';
import './index.css';

const App: React.FC = () => {
  const [threats, setThreats] = useState<Record<string, CameraStatus>>({
    cam1: { camera_id: 'cam1', violence: false, confidence: 0 },
    cam2: { camera_id: 'cam2', violence: false, confidence: 0 },
    cam3: { camera_id: 'cam3', violence: false, confidence: 0 },
    cam4: { camera_id: 'cam4', violence: false, confidence: 0 },
  });

  const [latestMessage, setLatestMessage] = useState<WebSocketMessage | null>(null);

  // WebSocket URL
  // Docker Desktop port forwards localhost:8000 to container backend:8000
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//localhost:8000/ws/threats`;

  const { isConnected } = useWebSocket({
    url: wsUrl,
    onMessage: (message: WebSocketMessage) => {
      setLatestMessage(message);

      if (message.type === 'threat_detection' && message.camera_id) {
        const cameraId = message.camera_id;
        setThreats((prev: Record<string, CameraStatus>) => ({
          ...prev,
          [cameraId]: {
            camera_id: cameraId,
            violence: message.violence ?? false,
            confidence: message.confidence ?? 0,
            timestamp: message.timestamp,
          },
        }));
      } else if (message.type === 'threat_status' && message.threats) {
        setThreats(message.threats);
      }
    },
    onError: (error: Event) => {
      console.error('WebSocket error:', error);
    },
  });

  return (
    <div className="w-screen h-screen bg-gray-950 flex flex-col text-white">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-white">Violence Detection Dashboard</h1>
            <p className="text-gray-400 text-sm mt-1">Real-time monitoring of 4 cameras</p>
          </div>
          <div className="flex items-center gap-3">
            <div
              className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}
            />
            <span className="text-sm">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Alert Banner */}
      <div className="bg-gray-900 border-b border-gray-800 px-4 py-3">
        <AlertBanner message={latestMessage} />
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-4">
        <div className="max-w-7xl mx-auto h-full">
          <CameraGrid threats={threats} />
        </div>
      </div>

      {/* Footer */}
      <div className="bg-gray-900 border-t border-gray-800 p-3 text-center text-sm text-gray-400">
        <p>
          {Object.values(threats).filter((t) => t.violence).length} threat(s) detected |
          Last update: {latestMessage?.timestamp ? new Date((latestMessage.timestamp ?? 0) * 1000).toLocaleTimeString() : 'N/A'}
        </p>
      </div>
    </div>
  );
};

export default App;
