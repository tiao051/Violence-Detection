/**
 * DetectionOverlay Component - Renders real-time threat status on video feed.
 */

import React, { useEffect, useRef } from 'react';
import { CameraStatus } from '../types/detection';

interface DetectionOverlayProps {
  cameraId: string;
  threat?: CameraStatus;
}

export const DetectionOverlay: React.FC<DetectionOverlayProps> = ({ threat }: DetectionOverlayProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !threat) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Only draw if violence detected
    if (!threat.violence) return;

    // Draw threat border
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.strokeRect(0, 0, canvas.width, canvas.height);

    // Draw confidence text
    const text = `Confidence: ${(threat.confidence * 100).toFixed(1)}%`;
    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 16px Arial';
    ctx.fillText(text, 10, 30);
  }, [threat]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute top-0 left-0 w-full h-full pointer-events-none"
      style={{ display: threat?.violence ? 'block' : 'none' }}
    />
  );
};

export default DetectionOverlay;
