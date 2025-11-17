/**
 * Type definitions for threat detection system.
 */

export interface IDetectionData {
  violence: boolean;
  confidence: number;
  class_id?: number;
  buffer_size?: number;
}

export interface IThreatAlert {
  camera_id: string;
  violence: boolean;
  confidence: number;
  timestamp: number;
}

export type WebSocketMessageType = 'threat_detection' | 'threat_status';

export interface IWebSocketMessage {
  type: WebSocketMessageType;
  camera_id?: string;
  violence?: boolean;
  confidence?: number;
  timestamp?: number;
  threats?: Record<string, IThreatAlert>;
}

export interface ICameraStatus {
  camera_id: string;
  violence: boolean;
  confidence: number;
  timestamp?: number;
}
