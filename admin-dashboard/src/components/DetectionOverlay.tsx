/**
 * DetectionOverlay Component - Renders real-time threat status on video feed.
 */

import React, { useCallback, useEffect, useRef } from 'react';
import { ICameraStatus } from '../types/detection';

interface IDetectionOverlayProps {
  cameraId: string;
  threat?: ICameraStatus;
}

interface ICanvasDrawConfig {
  strokeColor: string;
  fillColor: string;
  lineWidth: number;
  fontSize: string;
  textPadding: { x: number; y: number };
}

const CANVAS_CONFIG: ICanvasDrawConfig = {
  strokeColor: '#ef4444',
  fillColor: '#ef4444',
  lineWidth: 3,
  fontSize: 'bold 16px Arial',
  textPadding: { x: 10, y: 30 },
};

export const DetectionOverlay: React.FC<IDetectionOverlayProps> = ({ threat }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const clearCanvas = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number): void => {
    ctx.clearRect(0, 0, width, height);
  }, []);

  const drawThreatBorder = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number): void => {
    ctx.strokeStyle = CANVAS_CONFIG.strokeColor;
    ctx.lineWidth = CANVAS_CONFIG.lineWidth;
    ctx.strokeRect(0, 0, width, height);
  }, []);

  const drawConfidenceText = useCallback((ctx: CanvasRenderingContext2D, confidence: number): void => {
    const text = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    ctx.fillStyle = CANVAS_CONFIG.fillColor;
    ctx.font = CANVAS_CONFIG.fontSize;
    ctx.fillText(text, CANVAS_CONFIG.textPadding.x, CANVAS_CONFIG.textPadding.y);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !threat) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    clearCanvas(ctx, canvas.width, canvas.height);

    if (!threat.violence) {
      return;
    }

    drawThreatBorder(ctx, canvas.width, canvas.height);
    drawConfidenceText(ctx, threat.confidence);
  }, [threat, clearCanvas, drawThreatBorder, drawConfidenceText]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute top-0 left-0 w-full h-full pointer-events-none"
      style={{ display: threat?.violence ? 'block' : 'none' }}
    />
  );
};

export default DetectionOverlay;
