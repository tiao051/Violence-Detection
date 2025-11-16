/**
 * AlertBanner Component - Displays latest threat alert with severity indicator.
 */

import React from 'react';
import { WebSocketMessage } from '../types/detection';

interface AlertBannerProps {
  message?: WebSocketMessage | null;
}

export const AlertBanner: React.FC<AlertBannerProps> = ({ message }: AlertBannerProps) => {
  if (!message || !message.violence) {
    return (
      <div className="bg-green-900 bg-opacity-70 border-l-4 border-green-500 p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <span className="text-green-400 text-2xl">✓</span>
          </div>
          <div className="ml-3">
            <p className="text-green-200 font-semibold">All Systems Normal</p>
            <p className="text-green-300 text-sm">No threats detected</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-red-900 bg-opacity-90 border-l-4 border-red-500 p-4 animate-pulse">
      <div className="flex items-center">
        <div className="flex-shrink-0">
          <span className="text-red-400 text-2xl">⚠️</span>
        </div>
        <div className="ml-3">
          <p className="text-red-200 font-bold text-lg">VIOLENCE DETECTED!</p>
          <p className="text-red-300 text-sm">
            Camera: <span className="font-semibold">{message.camera_id?.toUpperCase()}</span> | 
            Confidence: <span className="font-semibold">{((message.confidence ?? 0) * 100).toFixed(1)}%</span>
          </p>
          <p className="text-red-300 text-xs mt-1">
            Timestamp: {new Date((message.timestamp ?? 0) * 1000).toLocaleTimeString()}
          </p>
        </div>
      </div>
    </div>
  );
};

export default AlertBanner;
