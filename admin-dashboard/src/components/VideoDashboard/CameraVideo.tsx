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
  const [isConnecting, setIsConnecting] = useState(false);

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
    setIsConnecting(false);
  }, []);

  const startWebRTC = useCallback(async () => {
    // Prevent multiple simultaneous connections
    if (isConnecting || pcRef.current) {
      return;
    }

    setIsConnecting(true);
    try {
      setError(null);
      setConnectionState("connecting");

      // Create optimized RTCPeerConnection for low latency
      const peer = new RTCPeerConnection({
        iceServers: [
          { urls: "stun:stun.l.google.com:19302" },
          { urls: "stun:stun1.l.google.com:19302" }
        ],
        bundlePolicy: "max-bundle",
        rtcpMuxPolicy: "require",
        iceTransportPolicy: "all"
      });

      pcRef.current = peer;

      // Add transceiver to limit bandwidth and configure direction
      peer.addTransceiver('video', {
        direction: 'recvonly'
      });

      // Monitor connection state
      peer.onconnectionstatechange = () => {
        setConnectionState(peer.connectionState);
        if (peer.connectionState === "failed" || peer.connectionState === "disconnected") {
          setError("Connection lost, reconnecting...");
          setIsConnecting(false);
          // Auto reconnect after 2 seconds
          reconnectTimeoutRef.current = setTimeout(() => {
            cleanup();
            startWebRTC();
          }, 2000);
        } else if (peer.connectionState === "connected") {
          setError(null);
          setIsConnecting(false);
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
      
      await peer.setLocalDescription({ type: "offer", sdp: offer.sdp });

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
      setIsConnecting(false);

      // Retry connection after 5 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        startWebRTC();
      }, 5000);
    }
  }, [cameraId, signalingServer, isConnecting]);

  useEffect(() => {
    startWebRTC();

    return () => {
      cleanup();
    };
  }, []); // Empty dependency array - only run once on mount

  return (
    <div className="camera-video-container">
      {/* Camera label */}
      <div className="camera-label">
        <span className="camera-name">{cameraId.toUpperCase()}</span>
      </div>

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
