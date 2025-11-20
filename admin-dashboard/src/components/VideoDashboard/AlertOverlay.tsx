import React, { useEffect, useState } from 'react';
import './AlertOverlay.css';
import './AlertOverlay.css';

interface ThreatAlert {
  type: 'alert';
  camera_id: string;
  timestamp: string;
  confidence: number;
}

interface AlertOverlayProps {
  alerts: ThreatAlert[];
  cameraId: string;
}

const AlertOverlay: React.FC<AlertOverlayProps> = ({ alerts, cameraId }) => {
  const [showAlert, setShowAlert] = useState(false);
  const [currentAlert, setCurrentAlert] = useState<ThreatAlert | null>(null);

  useEffect(() => {
    // Find the latest alert for this camera
    const cameraAlerts = alerts.filter(alert => alert.camera_id === cameraId);
    const latestAlert = cameraAlerts[cameraAlerts.length - 1];

    if (latestAlert && !currentAlert) {
      // New alert detected
      setCurrentAlert(latestAlert);
      setShowAlert(true);

      // Auto hide after 5 seconds
      const timer = setTimeout(() => {
        setShowAlert(false);
        setCurrentAlert(null);
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [alerts, cameraId, currentAlert]);

  if (!showAlert || !currentAlert) {
    return null;
  }

  return (
    <div className="alert-overlay">
      <div className="alert-content">
        <div className="alert-icon">🚨</div>
        <div className="alert-text">
          <div className="alert-title">THREAT DETECTED</div>
          <div className="alert-confidence">
            Confidence: {(currentAlert.confidence * 100).toFixed(1)}%
          </div>
          <div className="alert-time">
            {new Date(currentAlert.timestamp).toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlertOverlay;