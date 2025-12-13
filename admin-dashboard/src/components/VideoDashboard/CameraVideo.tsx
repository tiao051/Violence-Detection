import React, { useEffect, useRef, useState, useCallback } from "react";

interface CameraVideoProps {
  cameraId: string;       // cam1, cam2, cam3, cam4
  signalingServer?: string; // MediaMTX WebRTC HTTP (default 8889)
  isAlerting?: boolean;   // New prop for alert state
  alertSnapshot?: string | null; // New prop for alert snapshot
  onClick?: () => void;   // New prop for click handler
  isExpanded?: boolean;   // New prop for expanded state
}

const CameraVideo: React.FC<CameraVideoProps> = ({
  cameraId,
  signalingServer = import.meta.env.VITE_MEDIAMTX_URL || "http://localhost:8889",
  isAlerting = false,
  alertSnapshot = null,
  onClick,
  isExpanded = false
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionState, setConnectionState] = useState<string>("connecting");
  const [isConnecting, setIsConnecting] = useState(false);
  
  // Replay Logic
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const [isReplayMode, setIsReplayMode] = useState(false);
  const [replayUrl, setReplayUrl] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);

  // Audio Alert Logic
  const [isMuted, setIsMuted] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);

  const playAlarmSound = useCallback(() => {
    if (isMuted) return;
    
    try {
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        
        const ctx = audioContextRef.current;
        if (ctx && ctx.state === 'suspended') {
            ctx.resume();
        }

        if (ctx) {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(880, ctx.currentTime); // A5
            osc.frequency.linearRampToValueAtTime(440, ctx.currentTime + 0.5); // Drop to A4
            
            gain.gain.setValueAtTime(0.1, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.5);

            osc.connect(gain);
            gain.connect(ctx.destination);
            
            osc.start();
            osc.stop(ctx.currentTime + 0.5);
        }
    } catch (e) {
        console.error("Audio play failed", e);
    }
  }, [isMuted]);

  // Trigger alarm when alerting
  useEffect(() => {
    let interval: number;
    if (isAlerting && !isMuted) {
        playAlarmSound(); // Play immediately
        interval = window.setInterval(playAlarmSound, 1000); // Repeat every second
    }
    return () => clearInterval(interval);
  }, [isAlerting, isMuted, playAlarmSound]);

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
      videoRef.current.src = "";
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (replayUrl) {
      URL.revokeObjectURL(replayUrl);
    }
    setIsConnecting(false);
  }, [replayUrl]);

  // Start recording stream for buffer
  const startRecording = (stream: MediaStream) => {
    try {
      // Check supported mime types
      const mimeType = MediaRecorder.isTypeSupported("video/webm; codecs=vp9") 
        ? "video/webm; codecs=vp9" 
        : "video/webm";

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          recordedChunksRef.current.push(e.data);
          // Keep last ~60 seconds (assuming 1 chunk per second)
          if (recordedChunksRef.current.length > 60) {
            recordedChunksRef.current.shift();
          }
        }
      };

      recorder.start(1000); // 1 second chunks
    } catch (e) {
      console.error("MediaRecorder error:", e);
    }
  };

  const handleReplay = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent collapsing
    if (!videoRef.current || recordedChunksRef.current.length === 0) return;

    // Create blob from buffer
    const blob = new Blob(recordedChunksRef.current, { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    setReplayUrl(url);
    setIsReplayMode(true);
    setIsPaused(false);

    // Switch source
    videoRef.current.srcObject = null;
    videoRef.current.src = url;
    
    // Wait for metadata to seek
    videoRef.current.onloadedmetadata = () => {
      if (videoRef.current) {
        const duration = videoRef.current.duration;
        // Check if duration is finite to avoid TypeError
        if (Number.isFinite(duration)) {
           videoRef.current.currentTime = Math.max(0, duration - 3);
        } else {
           // Fallback for infinite duration (common in WebM blobs from MediaRecorder)
           // Just play from start or current position
           videoRef.current.currentTime = 0;
        }
        videoRef.current.play().catch(console.error);
      }
    };
  };

  const enterLiveMode = useCallback(() => {
    if (!videoRef.current) return;

    setIsReplayMode(false);
    setIsPaused(false);
    if (replayUrl) {
      URL.revokeObjectURL(replayUrl);
      setReplayUrl(null);
    }

    // Restore live stream
    if (pcRef.current) {
      const senders = pcRef.current.getReceivers();
      const track = senders.find(r => r.track.kind === 'video')?.track;
      if (track) {
        const stream = new MediaStream([track]);
        videoRef.current.src = "";
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(console.error);
      }
    }
  }, [replayUrl]);

  // Reset to live when collapsed
  useEffect(() => {
    if (!isExpanded && isReplayMode) {
      enterLiveMode();
    }
  }, [isExpanded, isReplayMode, enterLiveMode]);

  const handleGoLive = (e: React.MouseEvent) => {
    e.stopPropagation();
    enterLiveMode();
  };

  const togglePause = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!videoRef.current) return;

    if (videoRef.current.paused) {
      videoRef.current.play();
      setIsPaused(false);
    } else {
      videoRef.current.pause();
      setIsPaused(true);
    }
  };

  const seek = (seconds: number) => {
    if (!videoRef.current) return;
    const current = videoRef.current.currentTime;
    const duration = videoRef.current.duration;
    
    if (Number.isFinite(duration)) {
        videoRef.current.currentTime = Math.max(0, Math.min(duration, current + seconds));
    } else {
        // If duration is infinite, just seek relative to current
        videoRef.current.currentTime = Math.max(0, current + seconds);
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    if (!isExpanded || !isReplayMode) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") seek(-3);
      if (e.key === "ArrowRight") seek(3);
      if (e.key === " ") {
        e.preventDefault();
        if (videoRef.current) {
           if (videoRef.current.paused) {
             videoRef.current.play();
             setIsPaused(false);
           } else {
             videoRef.current.pause();
             setIsPaused(true);
           }
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isExpanded, isReplayMode]);

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
        // SYNC STRATEGY: Add delay to video to match AI processing latency
        // AI processing takes ~1-2s. We buffer video by 2s so alerts appear in sync with the action.
        if (event.receiver) {
          // Modern API (Chrome 112+, milliseconds)
          // @ts-ignore
          if (event.receiver.jitterBufferTarget !== undefined) {
             // @ts-ignore
             event.receiver.jitterBufferTarget = 2000; // Delay 2000ms (2s)
          } 
          // Legacy API (Chrome proprietary, seconds)
          // @ts-ignore
          else if (event.receiver.playoutDelayHint !== undefined) {
             // @ts-ignore
             event.receiver.playoutDelayHint = 2.0; // Delay 2.0s
          }
        }

        if (videoRef.current && event.streams[0]) {
          const stream = event.streams[0];
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => console.warn("Autoplay blocked"));
          setLoading(false);
          
          // Start recording for replay buffer
          startRecording(stream);
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
    <div 
      className={`camera-video-container ${isAlerting ? 'alert-active' : ''} ${isExpanded ? 'expanded' : ''}`}
      onClick={onClick}
    >
      {/* Camera label */}
      <div className="camera-label">
        <span className="camera-name">{cameraId.toUpperCase()}</span>
      </div>

      {/* Audio Control */}
      <button 
        className={`audio-control-btn ${isMuted ? 'muted' : ''}`}
        onClick={(e) => {
            e.stopPropagation();
            setIsMuted(!isMuted);
        }}
        title={isMuted ? "Unmute Alert" : "Mute Alert"}
      >
        {isMuted ? (
            <span>🔇</span>
        ) : (
            <span>🔊</span>
        )}
      </button>

      {/* Alert Overlay */}
      {isAlerting && (
        <div className="alert-overlay">
          <span className="alert-icon">⚠️</span>
          <span className="alert-text">VIOLENCE DETECTED</span>
          {alertSnapshot && (
            <div className="alert-snapshot">
              <img src={alertSnapshot} alt="Violence Snapshot" />
            </div>
          )}
        </div>
      )}

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

      {/* Replay Controls - Only show when expanded */}
      {isExpanded && (
        <div className="replay-controls" onClick={(e) => e.stopPropagation()}>
          {!isReplayMode ? (
            <button className="control-btn primary" onClick={handleReplay} title="Replay last 3s">
              <svg viewBox="0 0 24 24" className="control-icon">
                <path d="M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z"/>
              </svg>
            </button>
          ) : (
            <>
              <button className="control-btn" onClick={() => seek(-3)} title="Rewind 3s (Left Arrow)">
                <svg viewBox="0 0 24 24" className="control-icon">
                  <path d="M11 18V6l-8.5 6 8.5 6zm.5-6l8.5 6V6l-8.5 6z"/>
                </svg>
              </button>
              
              <button className="control-btn primary" onClick={togglePause} title="Play/Pause (Space)">
                {isPaused ? (
                  <svg viewBox="0 0 24 24" className="control-icon"><path d="M8 5v14l11-7z"/></svg>
                ) : (
                  <svg viewBox="0 0 24 24" className="control-icon"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                )}
              </button>

              <button className="control-btn" onClick={() => seek(3)} title="Forward 3s (Right Arrow)">
                <svg viewBox="0 0 24 24" className="control-icon">
                  <path d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z"/>
                </svg>
              </button>

              <button className="control-btn live-btn" onClick={handleGoLive} title="Return to Live Stream">
                GO LIVE
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default CameraVideo;
