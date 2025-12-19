import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface Alert {
  id: string;
  timestamp: number;  // Must be detection timestamp from backend (not client time)
  camera_id: string;
  violence_score: number;
  image_base64?: string;
  is_reviewed?: boolean;
  video_url?: string;
}

interface AlertContextType {
  alerts: Alert[];
  addAlert: (alert: Omit<Alert, 'id' | 'is_reviewed'>) => void;  // timestamp required
  updateAlert: (cameraId: string, timestamp: number, updates: Partial<Alert>) => void;
  clearAlerts: () => void;
  markAsReviewed: (id: string) => void;
  unreadCount: number;
}

const AlertContext = createContext<AlertContextType | undefined>(undefined);

export const AlertProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [alerts, setAlerts] = useState<Alert[]>(() => {
    const saved = localStorage.getItem('violence-alerts');
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => {
    localStorage.setItem('violence-alerts', JSON.stringify(alerts));
  }, [alerts]);

  const addAlert = (newAlertData: Omit<Alert, 'id' | 'is_reviewed'>) => {
    const COOLDOWN_SECONDS = 30; // 30 seconds grouping window
    
    // Alert MUST include detection timestamp from backend
    if (typeof newAlertData.timestamp !== 'number') {
      console.error('Alert must include detection timestamp:', newAlertData);
      return;
    }

    setAlerts(prev => {
      // Find if there's a recent alert for this camera (within cooldown window)
      // Compare detection times in SECONDS (all timestamps from backend are Unix time in seconds)
      const existingIndex = prev.findIndex(a => 
        a.camera_id === newAlertData.camera_id && 
        (newAlertData.timestamp - a.timestamp) < COOLDOWN_SECONDS
      );

      if (existingIndex !== -1) {
        // Found a recent event. Check if we should update it with better evidence.
        const existingAlert = prev[existingIndex];
        
        // If the new frame has higher confidence, update the snapshot and score
        // This ensures we capture the "peak" of the violence event
        if (newAlertData.violence_score > existingAlert.violence_score) {
          const updatedAlerts = [...prev];
          updatedAlerts[existingIndex] = {
            ...existingAlert,
            violence_score: newAlertData.violence_score,
            image_base64: newAlertData.image_base64 || existingAlert.image_base64,
            // Keep original timestamp (detection timestamp from first alert)
            timestamp: existingAlert.timestamp,
          };
          return updatedAlerts;
        }
        // If new frame is lower confidence, ignore it (it's part of the same event)
        return prev;
      }

      // No recent event, create a new alert with detection timestamp
      const alert: Alert = {
        ...newAlertData,
        id: crypto.randomUUID(),
        is_reviewed: false,
        // timestamp comes from newAlertData (detection timestamp)
      };
      return [alert, ...prev].slice(0, 100); // Keep last 100
    });
  };

  const updateAlert = (cameraId: string, timestamp: number, updates: Partial<Alert>) => {
    setAlerts(prev => {
      // Find alert that matches camera and is close in time (within 30 seconds)
      // timestamp is in SECONDS (Unix time from backend)
      const index = prev.findIndex(a => 
        a.camera_id === cameraId && 
        Math.abs(a.timestamp - timestamp) < 30
      );

      if (index !== -1) {
        const newAlerts = [...prev];
        newAlerts[index] = { ...newAlerts[index], ...updates };
        return newAlerts;
      }
      return prev;
    });
  };

  const clearAlerts = () => setAlerts([]);

  const markAsReviewed = (id: string) => {
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, is_reviewed: true } : a));
  };

  const unreadCount = alerts.filter(a => !a.is_reviewed).length;

  return (
    <AlertContext.Provider value={{ alerts, addAlert, updateAlert, clearAlerts, markAsReviewed, unreadCount }}>
      {children}
    </AlertContext.Provider>
  );
};

export const useAlerts = () => {
  const context = useContext(AlertContext);
  if (!context) throw new Error('useAlerts must be used within AlertProvider');
  return context;
};
