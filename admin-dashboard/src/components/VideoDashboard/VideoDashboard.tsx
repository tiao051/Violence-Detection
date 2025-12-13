import React from "react";
import CameraVideo from "./CameraVideo";
import { useWebSocket } from "../../hooks/useWebSocket";
import { useAlerts } from "../../contexts";
import "./VideoDashboard.css";

const cameras = ["cam1", "cam2", "cam3", "cam4"];

const VideoDashboard: React.FC = () => {
  // Connect to WebSocket for threat alerts
  const { alerts, isConnected, error } = useWebSocket(
    `ws://localhost:8000/ws/threats`
  );

  // Global Alert History Context
  const { addAlert } = useAlerts();

  // State for active alerts per camera (camera_id -> timestamp)
  const [activeAlerts, setActiveAlerts] = React.useState<Record<string, number>>({});
  // State for alert snapshots (camera_id -> base64 string)
  const [alertSnapshots, setAlertSnapshots] = React.useState<Record<string, string>>({});
  
  // State for expanded camera
  const [expandedCamera, setExpandedCamera] = React.useState<string | null>(null);

  // Process incoming alerts
  React.useEffect(() => {
    if (alerts.length > 0) {
      const latestAlert = alerts[alerts.length - 1];
      
      if (latestAlert.violence && latestAlert.camera_id) {
        // 1. Add to Global History (Context)
        // @ts-ignore
        addAlert({
          camera_id: latestAlert.camera_id,
          // @ts-ignore
          violence_score: latestAlert.confidence || 0.9, // Fallback if missing
          // @ts-ignore
          image_base64: latestAlert.snapshot
        });

        // 2. Update active alerts state (Visual Red Border)
        setActiveAlerts(prev => ({
          ...prev,
          [latestAlert.camera_id]: Date.now()
        }));

        // 3. Update snapshot if available
        // @ts-ignore
        if (latestAlert.snapshot) {
          setAlertSnapshots(prev => ({
            ...prev,
            // @ts-ignore
            [latestAlert.camera_id]: latestAlert.snapshot
          }));
        }

        // Auto-clear alert after 5 seconds if no new alerts come in
        setTimeout(() => {
          setActiveAlerts(prev => {
            const newState = { ...prev };
            // Only clear if the alert is older than 4.5 seconds (debounce)
            if (Date.now() - newState[latestAlert.camera_id] > 4500) {
              delete newState[latestAlert.camera_id];
            }
            return newState;
          });
          
          // Clear snapshot after alert ends
          setAlertSnapshots(prev => {
             const newState = { ...prev };
             if (Date.now() - activeAlerts[latestAlert.camera_id] > 4500) {
                delete newState[latestAlert.camera_id];
             }
             return newState;
          });
        }, 5000);
      }
    }
  }, [alerts]); // Only runs when new alert arrives

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
