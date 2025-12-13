import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface Alert {
  id: string;
  timestamp: number;
  camera_id: string;
  violence_score: number;
  image_base64?: string;
  is_reviewed?: boolean;
}

interface AlertContextType {
  alerts: Alert[];
  addAlert: (alert: Omit<Alert, 'id' | 'timestamp' | 'is_reviewed'>) => void;
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

  const addAlert = (newAlertData: Omit<Alert, 'id' | 'timestamp' | 'is_reviewed'>) => {
    const COOLDOWN_MS = 30000; // 30 seconds grouping window
    const now = Date.now();

    setAlerts(prev => {
      // Find if there's a recent alert for this camera
      const existingIndex = prev.findIndex(a => 
        a.camera_id === newAlertData.camera_id && 
        (now - a.timestamp) < COOLDOWN_MS
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
            // We keep the original timestamp to show when the event STARTED
          };
          return updatedAlerts;
        }
        // If new frame is lower confidence, ignore it (it's part of the same event)
        return prev;
      }

      // No recent event, create a new alert
      const alert: Alert = {
        ...newAlertData,
        id: crypto.randomUUID(),
        timestamp: now,
        is_reviewed: false,
      };
      return [alert, ...prev].slice(0, 100); // Keep last 100
    });
  };

  const clearAlerts = () => setAlerts([]);

  const markAsReviewed = (id: string) => {
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, is_reviewed: true } : a));
  };

  const unreadCount = alerts.filter(a => !a.is_reviewed).length;

  return (
    <AlertContext.Provider value={{ alerts, addAlert, clearAlerts, markAsReviewed, unreadCount }}>
      {children}
    </AlertContext.Provider>
  );
};

export const useAlerts = () => {
  const context = useContext(AlertContext);
  if (!context) throw new Error('useAlerts must be used within AlertProvider');
  return context;
};
