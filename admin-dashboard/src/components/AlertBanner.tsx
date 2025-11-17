/**
 * AlertBanner Component - Displays latest threat alert with severity indicator.
 */

import React, { useCallback, useMemo } from 'react';
import { IWebSocketMessage } from '../types/detection';

interface IAlertBannerProps {
  message?: IWebSocketMessage | null;
}

interface IAlertConfig {
  icon: string;
  title: string;
  subtitle: string;
}

const SAFE_STATUS_CONFIG: IAlertConfig = {
  icon: '✓',
  title: 'All Systems Normal',
  subtitle: 'No threats detected',
};

export const AlertBanner: React.FC<IAlertBannerProps> = ({ message }) => {
  const formatConfidence = useCallback((confidence: number): string => {
    return `${(confidence * 100).toFixed(1)}%`;
  }, []);

  const formatTimestamp = useCallback((timestamp: number): string => {
    return new Date(timestamp * 1000).toLocaleTimeString();
  }, []);

  const hasThreat = useMemo(() => {
    return message?.violence ?? false;
  }, [message]);

  const renderSafeStatus = useCallback((): JSX.Element => {
    return (
      <div className="bg-green-900 bg-opacity-70 border-l-4 border-green-500 p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <span className="text-green-400 text-2xl" aria-label="Safe status icon">
              {SAFE_STATUS_CONFIG.icon}
            </span>
          </div>
          <div className="ml-3">
            <p className="text-green-200 font-semibold">{SAFE_STATUS_CONFIG.title}</p>
            <p className="text-green-300 text-sm">{SAFE_STATUS_CONFIG.subtitle}</p>
          </div>
        </div>
      </div>
    );
  }, []);

  const renderThreatAlert = useCallback((alertMessage: IWebSocketMessage): JSX.Element => {
    const cameraId = alertMessage.camera_id?.toUpperCase() ?? 'UNKNOWN';
    const confidence = formatConfidence(alertMessage.confidence ?? 0);
    const timestamp = formatTimestamp(alertMessage.timestamp ?? 0);

    return (
      <div className="bg-red-900 bg-opacity-90 border-l-4 border-red-500 p-4 animate-pulse">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <span className="text-red-400 text-2xl" role="alert" aria-label="Threat alert icon">
              ⚠️
            </span>
          </div>
          <div className="ml-3">
            <p className="text-red-200 font-bold text-lg">VIOLENCE DETECTED!</p>
            <p className="text-red-300 text-sm">
              Camera: <span className="font-semibold">{cameraId}</span> | 
              Confidence: <span className="font-semibold">{confidence}</span>
            </p>
            <p className="text-red-300 text-xs mt-1">
              Timestamp: {timestamp}
            </p>
          </div>
        </div>
      </div>
    );
  }, [formatConfidence, formatTimestamp]);

  if (!hasThreat || !message) {
    return renderSafeStatus();
  }

  return renderThreatAlert(message);
};

export default AlertBanner;
