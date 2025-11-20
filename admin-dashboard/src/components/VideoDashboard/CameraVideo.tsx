import React, { useEffect, useRef, useState, useCallback } from "react";

interface CameraVideoProps {
  cameraId: string;       // cam1, cam2, cam3, cam4
  signalingServer?: string; // MediaMTX WebRTC HTTP (default 8889)
}

const CameraVideo: React.FC<CameraVideoProps> = ({
  cameraId,
  signalingServer = import.meta.env.VITE_MEDIAMTX_URL || "http://localhost:8889"
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionState, setConnectionState] = useState<string>("connecting");

  const cleanup = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const startWebRTC = useCallback(async () => {
    try {
      setError(null);
      setConnectionState("connecting");

      // Create optimized RTCPeerConnection for low latency
      const peer = new RTCPeerConnection({
        iceServers: [
          { urls: "stun:stun.l.google.com:19302" },
          { urls: "stun:stun1.l.google.com:19302" }
        ],
        // Optimize for low latency
        bundlePolicy: "max-bundle",
        rtcpMuxPolicy: "require",
        iceTransportPolicy: "all"
      });

      pcRef.current = peer;

      // Monitor connection state
      peer.onconnectionstatechange = () => {
        setConnectionState(peer.connectionState);
        if (peer.connectionState === "failed" || peer.connectionState === "disconnected") {
          setError("Connection lost, reconnecting...");
          // Auto reconnect after 2 seconds
          reconnectTimeoutRef.current = setTimeout(() => {
            cleanup();
            startWebRTC();
          }, 2000);
        } else if (peer.connectionState === "connected") {
          setError(null);
        }
      };

      peer.ontrack = (event) => {
        if (videoRef.current && event.streams[0]) {
          videoRef.current.srcObject = event.streams[0];
          videoRef.current.play().catch(() => console.warn("Autoplay blocked"));
          setLoading(false);
        }
      };

      // Create offer with optimized settings
      const offer = await peer.createOffer({
        offerToReceiveAudio: false,
        offerToReceiveVideo: true
      });
      await peer.setLocalDescription(offer);

      // Send offer to MediaMTX WHIP endpoint
      const res = await fetch(`${signalingServer}/${cameraId}/whep`, {
        method: "POST",
        body: offer.sdp,
        headers: { "Content-Type": "application/sdp" }
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const answerSdp = await res.text();
      await peer.setRemoteDescription({ type: "answer", sdp: answerSdp });

    } catch (err) {
      console.error(`WebRTC error for ${cameraId}:`, err);
      setError(`Failed to connect: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setLoading(false);

      // Retry connection after 5 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        startWebRTC();
      }, 5000);
    }
  }, [cameraId, signalingServer, cleanup]);

  useEffect(() => {
    startWebRTC();

    return () => {
      cleanup();
    };
  }, [startWebRTC, cleanup]);

  return (
    <div className="camera-video-container">
      {loading && <div className="loading-overlay">Loading {cameraId}...</div>}
      {error && <div className="error-overlay">{error}</div>}

      {/* Connection status indicator */}
      <div className={`connection-status ${connectionState}`}>
        <span className="status-dot"></span>
        <span className="status-text">{connectionState}</span>
      </div>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="camera-video-element"
      />
    </div>
  );
};

export default CameraVideo;
