import React from "react";
import CameraVideo from "./CameraVideo";
import { useWebSocket } from "../../hooks/useWebSocket";
import "./VideoDashboard.css";

const cameras = ["cam1", "cam2", "cam3", "cam4"];

const VideoDashboard: React.FC = () => {
  // Connect to WebSocket for threat alerts
  const { alerts, isConnected, error } = useWebSocket(
    `ws://localhost:8000/ws/threats`
  );

  // Log alerts to console
  React.useEffect(() => {
    if (alerts.length > 0) {
      console.log('🚨 Threat Alert Received:', alerts[alerts.length - 1]);
    }
  }, [alerts]);

  console.log('WebSocket status:', { isConnected, error, alertsCount: alerts.length });

  return (
    <div className="video-dashboard-grid">
      {cameras.map((cam) => (
        <CameraVideo key={cam} cameraId={cam} />
      ))}
    </div>
  );
};

export default VideoDashboard;
