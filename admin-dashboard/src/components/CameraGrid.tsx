/**
` * CameraGrid Component - Displays 2x2 grid of camera feeds with detection overlays.
 * Uses WebRTC (WHEP) for low-latency streaming (< 1s).
 * Features: Double-click for fullscreen, zoom, and pan controls.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ICameraStatus } from '../types/detection';
import DetectionOverlay from './DetectionOverlay';

interface ICameraGridProps {
  threats: Record<string, ICameraStatus>;
}

interface IPanPosition {
  x: number;
  y: number;
}

interface IZoomConfig {
  min: number;
  max: number;
  step: number;
}

const CAMERA_IDS = ['cam1', 'cam2', 'cam3', 'cam4'] as const;
const ZOOM_CONFIG: IZoomConfig = { min: 1, max: 5, step: 0.2 };
const RETRY_DELAY_MS = 3000;
const ICE_SERVERS = [{ urls: 'stun:stun.l.google.com:19302' }];

export const CameraGrid: React.FC<ICameraGridProps> = ({ threats }) => {
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const peerConnectionsRef = useRef<Record<string, RTCPeerConnection>>({});
  
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [zoom, setZoom] = useState<number>(ZOOM_CONFIG.min);
  const [pan, setPan] = useState<IPanPosition>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<IPanPosition>({ x: 0, y: 0 });

  const getWhepUrl = useCallback((cameraId: string): string => {
    return `/webrtc/${cameraId}/whep`;
  }, []);

  const clampZoom = useCallback((value: number): number => {
    return Math.max(ZOOM_CONFIG.min, Math.min(ZOOM_CONFIG.max, value));
  }, []);

  const resetViewState = useCallback((): void => {
    setZoom(ZOOM_CONFIG.min);
    setPan({ x: 0, y: 0 });
  }, []);

  const setupWebRTC = useCallback(async (cameraId: string): Promise<void> => {
    const video = videoRefs.current[cameraId];
    if (!video) {
      return;
    }

    try {
      const peerConnection = new RTCPeerConnection({ iceServers: ICE_SERVERS });

      peerConnection.addTransceiver('video', { direction: 'recvonly' });

      peerConnection.ontrack = (event: RTCTrackEvent): void => {
        console.log(`[${cameraId}] Received track:`, event.track.kind);
        video.srcObject = event.streams[0];
      };

      peerConnection.onconnectionstatechange = (): void => {
        console.log(`[${cameraId}] Connection state:`, peerConnection.connectionState);
        if (peerConnection.connectionState === 'failed') {
          console.error(`[${cameraId}] Connection failed, retrying...`);
          setTimeout(() => setupWebRTC(cameraId), RETRY_DELAY_MS);
        }
      };

      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);

      const response = await fetch(getWhepUrl(cameraId), {
        method: 'POST',
        headers: { 'Content-Type': 'application/sdp' },
        body: offer.sdp ?? '',
      });

      if (!response.ok) {
        throw new Error(`WHEP request failed with status: ${response.status}`);
      }

      const answer = await response.text();
      await peerConnection.setRemoteDescription({ type: 'answer', sdp: answer });

      peerConnectionsRef.current[cameraId] = peerConnection;
      console.log(`[${cameraId}] WebRTC connection established`);
    } catch (error) {
      console.error(`[${cameraId}] WebRTC setup error:`, error);
      setTimeout(() => setupWebRTC(cameraId), RETRY_DELAY_MS);
    }
  }, [getWhepUrl]);

  useEffect(() => {
    CAMERA_IDS.forEach((cameraId) => {
      setupWebRTC(cameraId).catch((error) => {
        console.error(`Failed to setup WebRTC for ${cameraId}:`, error);
      });
    });

    return () => {
      CAMERA_IDS.forEach((cameraId) => {
        const peerConnection = peerConnectionsRef.current[cameraId];
        if (peerConnection) {
          peerConnection.close();
          delete peerConnectionsRef.current[cameraId];
        }
      });
    };
  }, [setupWebRTC]);

  const handleCameraSelect = useCallback((cameraId: string): void => {
    setSelectedCamera(cameraId);
    resetViewState();
  }, [resetViewState]);

  const handleExitFullscreen = useCallback((): void => {
    setSelectedCamera(null);
    resetViewState();
  }, [resetViewState]);

  const handleZoomIn = useCallback((): void => {
    setZoom((prevZoom) => clampZoom(prevZoom + ZOOM_CONFIG.step));
  }, [clampZoom]);

  const handleZoomOut = useCallback((): void => {
    setZoom((prevZoom) => clampZoom(prevZoom - ZOOM_CONFIG.step));
  }, [clampZoom]);

  const handleWheel = useCallback((event: React.WheelEvent<HTMLDivElement>): void => {
    if (!selectedCamera) {
      return;
    }

    event.preventDefault();
    const delta = -event.deltaY * 0.001;
    setZoom((prevZoom) => clampZoom(prevZoom + delta));
  }, [selectedCamera, clampZoom]);

  const handleMouseDown = useCallback((event: React.MouseEvent<HTMLDivElement>): void => {
    if (selectedCamera && zoom > ZOOM_CONFIG.min) {
      setIsDragging(true);
      setDragStart({ x: event.clientX - pan.x, y: event.clientY - pan.y });
    }
  }, [selectedCamera, zoom, pan]);

  const handleMouseMove = useCallback((event: React.MouseEvent<HTMLDivElement>): void => {
    if (isDragging && selectedCamera && zoom > ZOOM_CONFIG.min) {
      setPan({
        x: event.clientX - dragStart.x,
        y: event.clientY - dragStart.y,
      });
    }
  }, [isDragging, selectedCamera, zoom, dragStart]);

  const handleMouseUp = useCallback((): void => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent): void => {
      if (!selectedCamera) {
        return;
      }

      switch (event.key) {
        case 'Escape':
          handleExitFullscreen();
          break;
        case '+':
        case '=':
          handleZoomIn();
          break;
        case '-':
        case '_':
          handleZoomOut();
          break;
        case '0':
          resetViewState();
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [selectedCamera, handleExitFullscreen, handleZoomIn, handleZoomOut, resetViewState]);

  return (
    <>
      {/* Grid View */}
      <div className={`grid grid-cols-2 gap-4 w-full h-full ${selectedCamera ? 'hidden' : ''}`}>
        {CAMERA_IDS.map((cameraId) => {
          const currentThreat = threats[cameraId];

          return (
            <div 
              key={cameraId} 
              className="relative bg-gray-900 rounded-lg overflow-hidden shadow-lg cursor-pointer hover:ring-4 hover:ring-blue-500 transition-all"
              onClick={() => handleCameraSelect(cameraId)}
            >
              <video
                ref={(el) => {
                  if (el) {
                    videoRefs.current[cameraId] = el;
                  }
                }}
                className="w-full h-full object-cover"
                autoPlay
                muted
                playsInline
              />

              <DetectionOverlay cameraId={cameraId} threat={currentThreat} />

              <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white px-3 py-1 rounded text-sm font-semibold">
                {cameraId.toUpperCase()}
              </div>

              <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs opacity-0 hover:opacity-100 transition-opacity">
                Click for fullscreen
              </div>

              {currentThreat?.violence && (
                <div className="absolute top-2 right-2 bg-red-600 text-white px-3 py-1 rounded text-xs font-bold animate-pulse">
                  ⚠️ VIOLENCE DETECTED
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Fullscreen View - Reuse same video element */}
      {selectedCamera && (
        <div 
          className="fixed inset-0 bg-black z-50 flex items-center justify-center"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <button
            onClick={handleExitFullscreen}
            className="absolute top-4 right-4 z-10 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-semibold"
            aria-label="Close fullscreen"
          >
            ✕ Close (ESC)
          </button>

          <div className="absolute bottom-4 right-4 z-10 flex flex-col gap-2">
            <button
              onClick={handleZoomIn}
              className="bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-bold text-xl"
              aria-label="Zoom in"
            >
              +
            </button>
            <div className="bg-gray-800 text-white px-3 py-1 rounded text-center text-sm">
              {zoom.toFixed(1)}x
            </div>
            <button
              onClick={handleZoomOut}
              className="bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-bold text-xl"
              aria-label="Zoom out"
            >
              −
            </button>
            <button
              onClick={resetViewState}
              className="bg-gray-800 hover:bg-gray-700 text-white px-3 py-1 rounded text-xs"
              aria-label="Reset zoom"
            >
              Reset
            </button>
          </div>

          <div className="relative w-full h-full overflow-hidden flex items-center justify-center">
            <div
              style={{
                transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                transition: isDragging ? 'none' : 'transform 0.2s ease-out',
                cursor: zoom > ZOOM_CONFIG.min ? (isDragging ? 'grabbing' : 'grab') : 'default',
              }}
              className="w-full h-full flex items-center justify-center"
            >
              <video
                className="w-full h-full object-contain"
                autoPlay
                muted
                playsInline
                ref={(el) => {
                  if (el) {
                    const gridVideo = videoRefs.current[selectedCamera];
                    if (gridVideo && gridVideo.srcObject) {
                      if (el.srcObject !== gridVideo.srcObject) {
                        el.srcObject = gridVideo.srcObject;
                      }
                    }
                  }
                }}
              />
            </div>

            <DetectionOverlay cameraId={selectedCamera} threat={threats[selectedCamera]} />

            <div className="absolute top-4 left-4 bg-black bg-opacity-70 text-white px-4 py-2 rounded text-lg font-semibold">
              {selectedCamera.toUpperCase()}
            </div>

            {threats[selectedCamera]?.violence && (
              <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-red-600 text-white px-6 py-3 rounded-lg text-lg font-bold animate-pulse">
                ⚠️ VIOLENCE DETECTED
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default CameraGrid;
