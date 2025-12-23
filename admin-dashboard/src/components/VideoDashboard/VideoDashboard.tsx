import React from "react";
import CameraVideo from "./CameraVideo";
import { useWebSocket, WebSocketMessage } from "../../hooks/useWebSocket";
import { useAlerts } from "../../contexts";
import "./VideoDashboard.css";

const cameras = ["cam1", "cam2", "cam3", "cam4"];

const VideoDashboard: React.FC = () => {
  // Connect to WebSocket for threat alerts
  const { messages, isConnected, error } = useWebSocket(
    `ws://localhost:8000/ws/threats`
  );

  // Global Alert History Context (Firestore-first)
  const { addOrUpdateEvent } = useAlerts();

  // State for active alerts per camera (camera_id -> { timestamp, eventId })
  const [activeAlerts, setActiveAlerts] = React.useState<Record<string, { timestamp: number; eventId?: string }>>({});
  // State for alert snapshots (camera_id -> base64 string)
  const [alertSnapshots, setAlertSnapshots] = React.useState<Record<string, string>>({});

  // State for expanded camera
  const [expandedCamera, setExpandedCamera] = React.useState<string | null>(null);

  // Track processed message count to avoid reprocessing
  const processedCountRef = React.useRef(0);

  // Process incoming messages (Firestore-first design, no severity)
  React.useEffect(() => {
    if (messages.length <= processedCountRef.current) return;

    // Process only new messages
    const newMessages = messages.slice(processedCountRef.current);
    processedCountRef.current = messages.length;

    newMessages.forEach((msg: WebSocketMessage) => {
      const { type, camera_id, timestamp, confidence, raw_confidence, snapshot, video_url, event_id, status } = msg;

      // Handle Firestore-first event messages (no severity)
      if ((type === 'event_started' || type === 'event_updated' || type === 'event_completed') && event_id) {
        // Update global alert history (synced with Firestore)
        addOrUpdateEvent({
          event_id,
          camera_id,
          timestamp: timestamp || Date.now() / 1000,
          confidence: confidence || 0,
          raw_confidence: raw_confidence,
          snapshot,
          video_url,
          status: status || 'active'
        });

        // Visual feedback for active alerts
        if (type === 'event_started' || type === 'event_updated') {

          setActiveAlerts(prev => ({
            ...prev,
            [camera_id]: { timestamp: Date.now(), eventId: event_id }
          }));

          if (snapshot) {
            setAlertSnapshots(prev => ({
              ...prev,
              [camera_id]: snapshot
            }));
          }

          // Auto-clear visual alert after 5 seconds
          setTimeout(() => {
            setActiveAlerts(prev => {
              const newState = { ...prev };
              if (newState[camera_id] && Date.now() - newState[camera_id].timestamp > 4500) {
                delete newState[camera_id];
              }
              return newState;
            });

            setAlertSnapshots(prev => {
              const newState = { ...prev };
              delete newState[camera_id];
              return newState;
            });
          }, 5000);
        }
      }
      // Legacy support for raw 'alert' type (from AI service directly)
      else if (type === 'alert' && msg.violence && camera_id) {
        // Visual feedback only (no add to history - wait for event_started)
        setActiveAlerts(prev => ({
          ...prev,
          // Preserve eventId if we already have one for this camera
          [camera_id]: {
            timestamp: Date.now(),
            eventId: prev[camera_id]?.eventId
          }
        }));

        if (snapshot) {
          setAlertSnapshots(prev => ({
            ...prev,
            [camera_id]: snapshot
          }));
        }

        setTimeout(() => {
          setActiveAlerts(prev => {
            const newState = { ...prev };
            if (newState[camera_id] && Date.now() - newState[camera_id].timestamp > 4500) {
              delete newState[camera_id];
            }
            return newState;
          });
        }, 5000);
      }
    });
  }, [messages, addOrUpdateEvent]);

  const handleCameraClick = (cameraId: string) => {
    if (expandedCamera === cameraId) {
      setExpandedCamera(null); // Collapse if already expanded
    } else {
      setExpandedCamera(cameraId); // Expand clicked camera
    }
  };

  return (
    <div className={`video-dashboard-grid ${expandedCamera ? 'has-expanded' : ''}`}>
      {cameras.map((cam) => (
        <CameraVideo
          key={cam}
          cameraId={cam}
          isAlerting={!!activeAlerts[cam]}
          alertSnapshot={alertSnapshots[cam]}
          alertId={activeAlerts[cam]?.eventId}
          isExpanded={expandedCamera === cam}
          onClick={() => handleCameraClick(cam)}
        />
      ))}
    </div>
  );
};

export default VideoDashboard;
