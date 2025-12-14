import React from "react";
import CameraVideo from "./CameraVideo";
import { useWebSocket } from "../../hooks/useWebSocket";
import { useAlerts } from "../../contexts";
import "./VideoDashboard.css";

const cameras = ["cam1", "cam2", "cam3", "cam4"];

const VideoDashboard: React.FC = () => {
  // Connect to WebSocket for threat alerts
  const { messages, isConnected, error } = useWebSocket(
    `ws://localhost:8000/ws/threats`
  );

  // Global Alert History Context
  const { addAlert, updateAlert } = useAlerts();

  // State for active alerts per camera (camera_id -> timestamp)
  const [activeAlerts, setActiveAlerts] = React.useState<Record<string, number>>({});
  // State for alert snapshots (camera_id -> base64 string)
  const [alertSnapshots, setAlertSnapshots] = React.useState<Record<string, string>>({});
  
  // State for expanded camera
  const [expandedCamera, setExpandedCamera] = React.useState<string | null>(null);

  // Process incoming messages
  React.useEffect(() => {
    if (messages.length > 0) {
      const latestMsg = messages[messages.length - 1];
      
      if (latestMsg.type === 'alert' && latestMsg.violence && latestMsg.camera_id) {
        // Validate that message has detection timestamp
        if (!latestMsg.timestamp) {
          console.warn('Alert message missing detection timestamp:', latestMsg);
          return;
        }

        // 1. Add to Global History (Context) with detection timestamp
        addAlert({
          camera_id: latestMsg.camera_id,
          violence_score: latestMsg.confidence || 0.9,
          image_base64: latestMsg.snapshot,
          timestamp: latestMsg.timestamp  // Detection timestamp from WebSocket
        });

        // 2. Update active alerts state (Visual Red Border)
        setActiveAlerts(prev => ({
          ...prev,
          [latestMsg.camera_id]: Date.now()
        }));

        // 3. Update snapshot if available
        if (latestMsg.snapshot) {
          setAlertSnapshots(prev => ({
            ...prev,
            [latestMsg.camera_id]: latestMsg.snapshot
          }));
        }

        // Auto-clear alert after 5 seconds
        setTimeout(() => {
          setActiveAlerts(prev => {
            const newState = { ...prev };
            if (Date.now() - newState[latestMsg.camera_id] > 4500) {
              delete newState[latestMsg.camera_id];
            }
            return newState;
          });
          
          setAlertSnapshots(prev => {
             const newState = { ...prev };
             if (Date.now() - activeAlerts[latestMsg.camera_id] > 4500) {
                delete newState[latestMsg.camera_id];
             }
             return newState;
          });
        }, 5000);
      } else if (latestMsg.type === 'event_saved' && latestMsg.video_url) {
        // Handle video saved event
        updateAlert(latestMsg.camera_id, latestMsg.timestamp, {
          video_url: latestMsg.video_url
        });
      }
    }
  }, [messages]); // Only runs when new message arrives

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
          isExpanded={expandedCamera === cam}
          onClick={() => handleCameraClick(cam)}
        />
      ))}
    </div>
  );
};

export default VideoDashboard;
