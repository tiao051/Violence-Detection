/**
 * CameraGrid Component - Displays 2x2 grid of camera feeds with detection overlays.
 * Uses WebRTC (WHEP) for low-latency streaming (< 1s).
 */

import React, { useEffect, useRef } from 'react';
import { CameraStatus } from '../types/detection';
import DetectionOverlay from './DetectionOverlay';

interface CameraGridProps {
  threats: Record<string, CameraStatus>;
}

export const CameraGrid: React.FC<CameraGridProps> = ({ threats }: CameraGridProps) => {
  const cameraIds = ['cam1', 'cam2', 'cam3', 'cam4'];
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const peerConnectionsRef = useRef<Record<string, RTCPeerConnection>>({});

  // Get WHEP URL for camera
  const getWhepUrl = (cameraId: string) => `/webrtc/${cameraId}/whep`;

  // Initialize WebRTC for each camera
  useEffect(() => {
    const setupWebRTC = async (cameraId: string) => {
      const video = videoRefs.current[cameraId];
      if (!video) return;

      try {
        // Create RTCPeerConnection
        const pc = new RTCPeerConnection({
          iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
        });

        // Add transceiver for receiving video
        pc.addTransceiver('video', { direction: 'recvonly' });

        // Handle incoming tracks
        pc.ontrack = (event) => {
          console.log(`[${cameraId}] Received track:`, event.track.kind);
          video.srcObject = event.streams[0];
        };

        // Handle connection state
        pc.onconnectionstatechange = () => {
          console.log(`[${cameraId}] Connection state:`, pc.connectionState);
          if (pc.connectionState === 'failed') {
            console.error(`[${cameraId}] Connection failed, retrying...`);
            setTimeout(() => setupWebRTC(cameraId), 3000);
          }
        };

        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // Send offer to MediaMTX WHEP endpoint
        const response = await fetch(getWhepUrl(cameraId), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/sdp',
          },
          body: offer.sdp,
        });

        if (!response.ok) {
          throw new Error(`WHEP request failed: ${response.status}`);
        }

        // Get answer and set remote description
        const answer = await response.text();
        await pc.setRemoteDescription({
          type: 'answer',
          sdp: answer,
        });

        peerConnectionsRef.current[cameraId] = pc;
        console.log(`[${cameraId}] WebRTC connection established`);
      } catch (error) {
        console.error(`[${cameraId}] WebRTC setup error:`, error);
        // Retry after 3 seconds
        setTimeout(() => setupWebRTC(cameraId), 3000);
      }
    };

    // Setup WebRTC for all cameras
    cameraIds.forEach((cameraId) => {
      setupWebRTC(cameraId);
    });

    // Cleanup
    return () => {
      cameraIds.forEach((cameraId) => {
        const pc = peerConnectionsRef.current[cameraId];
        if (pc) {
          pc.close();
          delete peerConnectionsRef.current[cameraId];
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
            playsInline
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
