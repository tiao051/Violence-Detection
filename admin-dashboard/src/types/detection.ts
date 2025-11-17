/**
 * Type definitions for threat detection system.
 */

export interface DetectionData {
  violence: boolean;
  confidence: number;
  class_id?: number;
  buffer_size?: number;
}

export interface ThreatAlert {
  camera_id: string;
  violence: boolean;
  confidence: number;
  timestamp: number;
}

export interface WebSocketMessage {
  type: 'threat_detection' | 'threat_status';
  camera_id?: string;
  violence?: boolean;
  confidence?: number;
  timestamp?: number;
  threats?: Record<string, ThreatAlert>;
}

export interface CameraStatus {
  camera_id: string;
  violence: boolean;
  confidence: number;
  timestamp?: number;
}
