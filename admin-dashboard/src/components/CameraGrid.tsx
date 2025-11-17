/**
 * CameraGrid Component - Displays 2x2 grid of camera feeds with detection overlays.
 */

import React, { useEffect, useRef } from 'react';
import HLS from 'hls.js';
import { CameraStatus } from '../types/detection';
import DetectionOverlay from './DetectionOverlay';

interface CameraGridProps {
  threats: Record<string, CameraStatus>;
}

export const CameraGrid: React.FC<CameraGridProps> = ({ threats }: CameraGridProps) => {
  const cameraIds = ['cam1', 'cam2', 'cam3', 'cam4'];
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});

  // Get HLS URL for camera
  const getHlsUrl = (cameraId: string) => `/hls/${cameraId}/index.m3u8`;

  // Initialize HLS.js for each video
  useEffect(() => {
    cameraIds.forEach((cameraId) => {
      const video = videoRefs.current[cameraId];
      if (!video) return;

      // Destroy existing HLS instance
      const videoWithHls = video as any;
      if (videoWithHls.hls) {
        videoWithHls.hls.destroy();
      }

      if (HLS.isSupported()) {
        const hls = new HLS({
          debug: false,
          enableWorker: true,
          maxBufferSize: 60 * 1000 * 1000, // 60MB
          maxBufferLength: 30,
        });

        hls.loadSource(getHlsUrl(cameraId));
        hls.attachMedia(video);
        videoWithHls.hls = hls;

        hls.on(HLS.Events.MANIFEST_PARSED, () => {
          video.play().catch((err) => console.error(`Play error for ${cameraId}:`, err));
        });

        hls.on(HLS.Events.ERROR, (_, data) => {
          console.error(`HLS error for ${cameraId}:`, data);
        });
      } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        // Safari native HLS support
        video.src = getHlsUrl(cameraId);
        video.play().catch((err) => console.error(`Play error for ${cameraId}:`, err));
      }
    });

    // Cleanup
    return () => {
      cameraIds.forEach((cameraId) => {
        const video = videoRefs.current[cameraId];
        if (video) {
          const videoWithHls = video as any;
          if (videoWithHls.hls) {
            videoWithHls.hls.destroy();
          }
        }
      });
    };
  }, []);

  return (
    <div className="grid grid-cols-2 gap-4 w-full h-full">
      {cameraIds.map((cameraId) => (
        <div key={cameraId} className="relative bg-gray-900 rounded-lg overflow-hidden shadow-lg">
          {/* Camera Feed */}
          <video
            ref={(el) => {
              if (el) videoRefs.current[cameraId] = el;
            }}
            className="w-full h-full object-cover"
            autoPlay
            muted
            controls
          />

          {/* Detection Overlay */}
          <DetectionOverlay
            cameraId={cameraId}
            threat={threats[cameraId]}
          />

          {/* Camera Label */}
          <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white px-3 py-1 rounded text-sm font-semibold">
            {cameraId.toUpperCase()}
          </div>

          {/* Threat Status Badge */}
          {threats[cameraId]?.violence && (
            <div className="absolute top-2 right-2 bg-red-600 text-white px-3 py-1 rounded text-xs font-bold animate-pulse">
              ⚠️ VIOLENCE DETECTED
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default CameraGrid;
